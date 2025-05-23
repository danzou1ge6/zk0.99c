#include "../../mont/src/bn254_scalar.cuh"
#include "../../mont/src/bn254_fr.cuh"
#include "bn254.cuh"
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <array>
#include <chrono>
#include <cub/cub.cuh>
#include <iostream>
#include <thread>

#define PROPAGATE_CUDA_ERROR(x)                                                                                    \
{                                                                                                                \
    err = x;                                                                                                       \
    if (err != cudaSuccess)                                                                                        \
    {                                                                                                              \
        std::cerr << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: " << cudaGetErrorString(err) << std::endl; \
        return err;                                                                                                  \
    }                                                                                                              \
}

namespace msm {

    using mont::u32;
    using mont::u64;
    using mont::usize;

    const u32 THREADS_PER_WARP = 32;

    constexpr u32 pow2(u32 n) {
        return n == 0 ? 1 : 2 * pow2(n - 1);
    }

    constexpr __forceinline__ __host__ __device__ u32 div_ceil(u32 a, u32 b) {
        return (a + b - 1) / b;
    }

    constexpr int log2_floor(int n) {
      return (n == 1) ? 0 : 1 + log2_floor(n / 2);
    }

    constexpr int log2_ceil(int n) {
      // Check if n is a power of 2
        if ((n & (n - 1)) == 0)
            return log2_floor(n);
        else
          return 1 + log2_floor(n);
    }

    template <typename T, u32 D1, u32 D2>
    struct Array2D {
        T *buf;

        Array2D() {}
        Array2D(T *buf) : buf(buf) {}
        __host__ __device__ __forceinline__ T &get(u32 i, u32 j) {
          return buf[i * D2 + j];
        }
        __host__ __device__ __forceinline__ const T &get_const(u32 i, u32 j) const {
          return buf[i * D2 + j];
        }
        __host__ __device__ __forceinline__ T *addr(u32 i, u32 j) {
          return buf + i * D2 + j;
        }
    };

    template <u32 BITS = 255, u32 WINDOW_SIZE = 22,
    u32 TARGET_WINDOWS = 1, bool DEBUG = true>
    struct MsmConfig {
        static constexpr u32 lambda = BITS;
        static constexpr u32 s = WINDOW_SIZE; // must <= 31
        static constexpr u32 n_buckets = pow2(s - 1); // [1, 2^{s-1}] buckets, using signed digit to half the number of buckets

        // # of logical windows
        static constexpr u32 actual_windows = div_ceil(lambda, s);
        
        // stride for precomputation(same as # of windows), if >= actual_windows, no precomputation; if = 1, precompute all
        static constexpr u32 n_windows = actual_windows < TARGET_WINDOWS ? actual_windows : TARGET_WINDOWS;

        static constexpr u32 window_bits = log2_ceil(n_windows);
        
        // lines of points to be stored in memory, 1 for no precomputation
        static constexpr u32 n_precompute = div_ceil(actual_windows, n_windows);

        static constexpr bool debug = DEBUG;
    };

    template <u32 windows, u32 bits_per_window>
    __host__ __device__ __forceinline__ void signed_digit(int (&r)[windows]) {
        static_assert(bits_per_window < 32, "bits_per_window must be less than 32");
        #pragma unroll
        for (u32 i = 0; i < windows - 1; i++) {
            if ((u32)r[i] >= 1u << (bits_per_window - 1)) {
                r[i] -= 1 << bits_per_window;
                r[i + 1] += 1;
            }
        }
        assert((u32)r[windows - 1] < 1u << (bits_per_window - 1));
    }

    // divide scalers into windows
    // count number of zeros in each window
    template <typename Config, typename Number>
    __global__ void distribute_windows(
        const u32 *scalers,
        const u64 len,
        u32* cnt_zero,
        u64* indexs,
        u32* points_offset
    ) {
        u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
        u32 stride = gridDim.x * blockDim.x;
        
        // Count into block-wide counter
        for (u32 i = tid; i < len; i += stride) {
            int bucket[Config::actual_windows];
            auto scaler = Number::load(scalers + i * Number::LIMBS);
            scaler.bit_slice<Config::actual_windows, Config::s>(bucket);
            signed_digit<Config::actual_windows, Config::s>(bucket);

            #pragma unroll
            for (u32 window_id = 0; window_id < Config::actual_windows; window_id++) {
                auto sign = bucket[window_id] < 0;
                auto bucket_id = sign ? -bucket[window_id] : bucket[window_id];
                u32 physical_window_id = window_id % Config::n_windows;
                u32 point_group = window_id / Config::n_windows;
                if (bucket_id == 0) {
                    atomicAdd(cnt_zero, 1);
                }
                u64 index = bucket_id | (sign << Config::s) | (physical_window_id << (Config::s + 1)) 
                | ((point_group * len + i) << (Config::s + 1 + Config::window_bits));
                indexs[(points_offset[physical_window_id] + point_group) * len + i] = index;
            }
        }
    }

    __device__ __forceinline__ void lock(unsigned short *mutex_ptr, u32 wait_limit = 16) {
        u32 time = 1;
        while (atomicCAS(mutex_ptr, (unsigned short int)0, (unsigned short int)1) != 0) {
            __nanosleep(time);
            if likely(time < wait_limit) time *= 2;
        }
        __threadfence();
    }

    __device__ __forceinline__ void unlock(unsigned short *mutex_ptr) {
        __threadfence();
        atomicCAS(mutex_ptr, (unsigned short int)1, (unsigned short int)0);
    }

    template <typename Config, u32 WarpPerBlock, typename Point, typename PointAffine>
    __global__ void bucket_sum(
        const u64 len,
        const u32 *cnt_zero,
        const u64 *indexs,
        const u32 *points,
        Array2D<unsigned short, Config::n_windows, Config::n_buckets> mutex,
        Array2D<unsigned short, Config::n_windows, Config::n_buckets> initialized,
        Array2D<Point, Config::n_windows, Config::n_buckets> sum
    ) {
        __shared__ u32 point_buffer[WarpPerBlock * THREADS_PER_WARP][PointAffine::N_WORDS * 2 + 4]; // +4 for padding and alignment
        const static u32 key_mask = (1u << Config::s) - 1;
        const static u32 sign_mask = 1u << Config::s;
        const static u32 window_mask = (1u << Config::window_bits) - 1;

        const u32 gtid = threadIdx.x + blockIdx.x * blockDim.x;
        const u32 threads = blockDim.x * gridDim.x;
        const u32 zero_num = *cnt_zero;

        u32 work_len = div_ceil(len - zero_num, threads);
        const u32 start_id = work_len * gtid;
        const u32 end_id = min((u64)start_id + work_len, len - zero_num);
        if (start_id >= end_id) return;

        indexs += zero_num;

        int stage = 0;
        uint4 *smem_ptr0 = reinterpret_cast<uint4*>(point_buffer[threadIdx.x]);
        uint4 *smem_ptr1 = reinterpret_cast<uint4*>(point_buffer[threadIdx.x] + PointAffine::N_WORDS);

        bool first = true; // only the first bucket and the last bucket may have conflict with other threads
        auto pip_thread = cuda::make_pipeline(); // pipeline for this thread

        u64 index = indexs[start_id];
        u64 pointer = index >> (Config::s + 1 + Config::window_bits);
        u32 key = index & key_mask;
        u32 sign = (index & sign_mask) != 0;
        u32 window_id = (index >> (Config::s + 1)) & window_mask;
        
        // used to special handle the last bucket
        u64 last_index = indexs[end_id - 1];
        u32 last_key = last_index & key_mask;
        u32 last_window_id = (last_index >> (Config::s + 1)) & window_mask;

        pip_thread.producer_acquire();
        cuda::memcpy_async(smem_ptr0, reinterpret_cast<const uint4*>(points + pointer * PointAffine::N_WORDS), sizeof(PointAffine), pip_thread);
        stage ^= 1;
        pip_thread.producer_commit();

        auto acc = Point::identity();

        for (u32 i = start_id + 1; i < end_id; i++) {
            u64 next_index = indexs[i];
            pointer = next_index >> (Config::s + 1 + Config::window_bits);

            uint4 *g2s_ptr, *s2r_ptr;
            if (stage == 0) {
                g2s_ptr = smem_ptr0;
                s2r_ptr = smem_ptr1;
            } else {
                g2s_ptr = smem_ptr1;
                s2r_ptr = smem_ptr0;
            }

            pip_thread.producer_acquire();
            cuda::memcpy_async(g2s_ptr, reinterpret_cast<const uint4*>(points + pointer * PointAffine::N_WORDS), sizeof(PointAffine), pip_thread);
            pip_thread.producer_commit();
            stage ^= 1;
            
            cuda::pipeline_consumer_wait_prior<1>(pip_thread);
            auto p = PointAffine::load(reinterpret_cast<u32*>(s2r_ptr));
            pip_thread.consumer_release();

            if (sign) p = p.neg();
            acc = acc + p;

            u32 next_key = next_index & key_mask;
            u32 next_window_id = (next_index >> (Config::s + 1)) & window_mask;

            if unlikely(next_key != key || next_window_id != window_id) {
                if unlikely(first) {
                    unsigned short *mutex_ptr;
                    mutex_ptr = mutex.addr(window_id, key - 1);
                    lock(mutex_ptr);
                    if (initialized.get(window_id, key - 1)) {
                        sum.get(window_id, key - 1) = sum.get(window_id, key - 1) + acc;
                    } else {
                        sum.get(window_id, key - 1) = acc;
                        initialized.get(window_id, key - 1) = 1;
                    }
                    unlock(mutex_ptr);
                    first = false;
                } else {
                    sum.get(window_id, key - 1) = acc;
                    initialized.get(window_id, key - 1) = 1;
                }

                if (initialized.get(next_window_id, next_key - 1) && (next_key != last_key || next_window_id != last_window_id)) {
                    acc = sum.get(next_window_id, next_key - 1);
                } else {
                    acc = Point::identity();
                }
            }
            key = next_key;
            sign = (next_index & sign_mask) != 0;
            window_id = next_window_id;
        }

        pip_thread.consumer_wait();
        auto p = PointAffine::load(reinterpret_cast<u32*>(stage == 0 ? smem_ptr1 : smem_ptr0));
        pip_thread.consumer_release();

        if (sign) p = p.neg();
        acc = acc + p;
        
        auto mutex_ptr = mutex.addr(window_id, key - 1);
        lock(mutex_ptr);
        if (initialized.get(window_id, key - 1)) {
            sum.get(window_id, key - 1) = sum.get(window_id, key - 1) + acc;
        } else {
            sum.get(window_id, key - 1) = acc;
            initialized.get(window_id, key - 1) = 1;
        }
        unlock(mutex_ptr);
    }

    template<typename Config, u32 WarpPerBlock, typename Point>
    __launch_bounds__(256,1)
    __global__ void reduceBuckets(
        Array2D<Point, Config::n_windows, Config::n_buckets> buckets_sum, 
        Point *reduceMemory,
        Array2D<unsigned short, Config::n_windows, Config::n_buckets> initialized
    ) {

        assert(gridDim.x % Config::n_windows == 0);

        __shared__ u32 smem[WarpPerBlock][Point::N_WORDS + 4]; // +4 for padding

        const u32 total_threads_per_window = gridDim.x / Config::n_windows * blockDim.x;
        u32 window_id = blockIdx.x / (gridDim.x / Config::n_windows);

        u32 wtid = (blockIdx.x % (gridDim.x / Config::n_windows)) * blockDim.x + threadIdx.x;
          
        const u32 buckets_per_thread = div_ceil(Config::n_buckets, total_threads_per_window);

        Point sum, sum_of_sums;

        sum = Point::identity();
        sum_of_sums = Point::identity();

        for(u32 i=buckets_per_thread; i > 0; i--) {
            u32 loadIndex = wtid * buckets_per_thread + i;
            if(loadIndex <= Config::n_buckets && initialized.get(window_id, loadIndex - 1)) {
                sum = sum + buckets_sum.get(window_id, loadIndex - 1);
            }
            sum_of_sums = sum_of_sums + sum;
        }

        u32 scale = wtid * buckets_per_thread;

        sum = sum.multiple(scale);

        sum_of_sums = sum_of_sums + sum;

        // Reduce within the block
        // 1. reduce in each warp
        // 2. store to smem
        // 3. reduce in warp1
        u32 warp_id = threadIdx.x / 32;
        u32 lane_id = threadIdx.x % 32;

        sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(16);
        sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(8);
        sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(4);
        sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(2);
        sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(1);

        if (lane_id == 0) {
            sum_of_sums.store(smem[warp_id]);
        }

        __syncthreads();

        if (warp_id > 0) return;

        if (threadIdx.x < WarpPerBlock) {
            sum_of_sums = Point::load(smem[threadIdx.x]);
        } else {
            sum_of_sums = Point::identity();
        }

        // Reduce in warp1
        if constexpr (WarpPerBlock > 16) sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(16);
        if constexpr (WarpPerBlock > 8) sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(8);
        if constexpr (WarpPerBlock > 4) sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(4);
        if constexpr (WarpPerBlock > 2) sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(2);
        if constexpr (WarpPerBlock > 1) sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(1);

        // Store to global memory
        if (threadIdx.x == 0) {
            for (u32 i = 0; i < window_id * Config::s; i++) {
               sum_of_sums = sum_of_sums.self_add();
            }
            reduceMemory[blockIdx.x] = sum_of_sums;
        }
    }

    template <typename Config, typename Point, typename PointAffine>
    __global__ void precompute_kernel(u32 *points, u64 len) {
        u64 gid = threadIdx.x + blockIdx.x * blockDim.x;
        if (gid >= len) return;
        auto p = PointAffine::load(points + gid * PointAffine::N_WORDS).to_point();
        for (u32 i = 1; i < Config::n_precompute; i++) {
            #pragma unroll
            for (u32 j = 0; j < Config::n_windows * Config::s; j++) {
                p = p.self_add();
            }

            p.to_affine().store(points + (gid + i * len) * PointAffine::N_WORDS);
        }
    }

    template <typename Config, typename Number, typename Point, typename PointAffine>
    class MSM {
        int device;
        std::array<const u32*, Config::n_precompute> h_points;
        Point **d_buckets_sum_buf;
        Array2D<Point, Config::n_windows, Config::n_buckets> *buckets_sum;
        unsigned short *mutex_buf;
        unsigned short **initialized_buf;
        Array2D<unsigned short, Config::n_windows, Config::n_buckets> *initialized;
        // record for number of logical windows in each actual window
        u32 h_points_per_window[Config::n_windows];
        u32 h_points_offset[Config::n_windows + 1];
        u32 *d_points_offset;
        u32 *cnt_zero;
        u64 *indexs;
        // indexs is used to store the bucket id, point index and sign
        // Config::s bits for bucket id, 1 bit for sign, Config::window_bits for window id, the rest for point index
        // max log(2^30(max points) * precompute) + Config::s bits are needed
        static_assert(log2_ceil(Config::n_precompute) + 30 + Config::window_bits + Config::s + 1 <= 64, "Index too large");
        // for sorting, after sort, points with same bucket id are gathered, gives pointer to original index
        u32 **scalers;
        void *d_temp_storage_sort;
        usize temp_storage_bytes_sort;
        u32 **points;
        cudaEvent_t *begin_scaler_copy, *end_scaler_copy;
        cudaEvent_t *begin_point_copy, *end_point_copy;
        Point **reduce_buffer;
        Point **h_reduce_buffer;
        u32 num_sm, reduce_blocks;
        u32 stage_scaler, stage_point, stage_point_transporting;
        const u32 batch_per_run;
        const u32 parts;
        const u32 max_scaler_stages;
        const u32 max_point_stages;
        const u64 len;
        bool allocated = false;
        bool head = true;
        bool points_set = false;

        cudaError_t run(const u32 batches, std::vector<u32*>::const_iterator h_scalers, bool first_run, cudaStream_t stream) {
            cudaError_t err;

            u64 part_len = div_ceil(len, parts);

            for (u32 i = 0; i < batches; i++) {
                PROPAGATE_CUDA_ERROR(cudaMemsetAsync(initialized_buf[i], 0, sizeof(unsigned short) * (Config::n_buckets) * Config::n_windows, stream));
            }

            auto mutex = Array2D<unsigned short, Config::n_windows, Config::n_buckets>(mutex_buf);

            int begin = 0, end = parts, stride = 1;

            if (head) {
                begin = 0;
                end = parts;
                stride = 1;
                head = false;
            } else {
                begin = parts - 1;
                end = -1;
                stride = -1;
                head = true;
            }
            u32 first_len = std::min(part_len, len - begin * part_len);
            u32 points_transported = first_run ? 0 : first_len;
            
            cudaStream_t copy_stream;
            PROPAGATE_CUDA_ERROR(cudaStreamCreate(&copy_stream));
            for (u32 i = 0; i < max_scaler_stages; i++) {
                PROPAGATE_CUDA_ERROR(cudaEventRecord(begin_scaler_copy[i], stream));
            }
            for (u32 i = 0; i < max_point_stages; i++) {
                PROPAGATE_CUDA_ERROR(cudaEventRecord(begin_point_copy[i], stream));
            }
            
            u32 points_per_transfer;

            for (int p = begin; p != end; p += stride) {
                u64 offset = p * part_len;
                u64 cur_len = std::min(part_len, len - offset);
                if (p + stride != end) {
                    auto next_len = std::min(part_len, len - (p + stride) * part_len);
                    points_per_transfer = div_ceil(next_len, batches);
                }

                for (int j = 0; j < batches; j++) {
                    PROPAGATE_CUDA_ERROR(cudaStreamWaitEvent(copy_stream, begin_scaler_copy[stage_scaler], cudaEventWaitDefault));
                    PROPAGATE_CUDA_ERROR(cudaMemcpyAsync(scalers[stage_scaler], *(h_scalers + j) + offset * Number::LIMBS, sizeof(u32) * Number::LIMBS * cur_len, cudaMemcpyHostToDevice, copy_stream));
                    PROPAGATE_CUDA_ERROR(cudaEventRecord(end_scaler_copy[stage_scaler], copy_stream));

                    PROPAGATE_CUDA_ERROR(cudaMemsetAsync(cnt_zero, 0, sizeof(u32), stream));
                    
                    PROPAGATE_CUDA_ERROR(cudaStreamWaitEvent(stream, end_scaler_copy[stage_scaler], cudaEventWaitDefault));
                    cudaEvent_t start, stop;
                    float elapsedTime = 0.0;
                    cudaEventCreate(&start);
                    cudaEventCreate(&stop);
                    if constexpr (Config::debug) {
                        cudaEventRecord(start, stream);
                    }
                    
                    u32 block_size = 512;
                    u32 grid_size = num_sm;
                    distribute_windows<Config, Number><<<grid_size, block_size, 0, stream>>>(
                        scalers[stage_scaler],
                        cur_len,
                        cnt_zero,
                        indexs + part_len * Config::actual_windows,
                        d_points_offset
                    );

                    PROPAGATE_CUDA_ERROR(cudaGetLastError());
                    PROPAGATE_CUDA_ERROR(cudaEventRecord(begin_scaler_copy[stage_scaler], stream));

                    if constexpr (Config::debug) {
                        cudaEventRecord(stop, stream);
                        PROPAGATE_CUDA_ERROR(cudaEventSynchronize(stop));
                        cudaEventElapsedTime(&elapsedTime, start, stop);
                        std::cout << "MSM distribute_windows time:" << elapsedTime << std::endl;
                    }

                    if constexpr (Config::debug) {
                        cudaEventRecord(start, stream);
                    }

                    cub::DeviceRadixSort::SortKeys(
                        d_temp_storage_sort, temp_storage_bytes_sort,
                        indexs + part_len * Config::actual_windows, indexs,
                        Config::actual_windows * cur_len, 0, Config::s, stream
                    );

                    PROPAGATE_CUDA_ERROR(cudaGetLastError());
                    if constexpr (Config::debug) {
                        cudaEventRecord(stop, stream);
                        PROPAGATE_CUDA_ERROR(cudaEventSynchronize(stop));
                        cudaEventElapsedTime(&elapsedTime, start, stop);
                        std::cout << "MSM sort time:" << elapsedTime << std::endl;
                    }

                    // wait before the first point copy
                    if (points_transported == 0) PROPAGATE_CUDA_ERROR(cudaStreamWaitEvent(copy_stream, begin_point_copy[stage_point_transporting], cudaEventWaitDefault));
                    
                    if (j == 0) {
                        u32 point_left = cur_len - points_transported;
                        if (point_left > 0) for (int i = 0; i < Config::n_precompute; i++) {
                            PROPAGATE_CUDA_ERROR(cudaMemcpyAsync(
                                points[stage_point_transporting] + (i * cur_len + points_transported) * PointAffine::N_WORDS,
                                h_points[i] + (offset + points_transported) * PointAffine::N_WORDS,
                                sizeof(PointAffine) * point_left, cudaMemcpyHostToDevice, copy_stream
                            ));
                        }
                        PROPAGATE_CUDA_ERROR(cudaEventRecord(end_point_copy[stage_point_transporting], copy_stream));
                        points_transported = 0;
                        if (p + stride != end) stage_point_transporting = (stage_point_transporting + 1) % max_point_stages;
                    } else if(p + stride != end) {
                        u64 next_offset = (p + stride) * part_len;

                        for (int i = 0; i < Config::n_precompute; i++) {
                            PROPAGATE_CUDA_ERROR(cudaMemcpyAsync(
                                points[stage_point_transporting] + (i * cur_len + points_transported) * PointAffine::N_WORDS,
                                h_points[i] + (next_offset + points_transported) * PointAffine::N_WORDS,
                                sizeof(PointAffine) * points_per_transfer, cudaMemcpyHostToDevice, copy_stream
                            ));
                        }
                        points_transported += points_per_transfer;
                    }

                    if (j == 0) PROPAGATE_CUDA_ERROR(cudaStreamWaitEvent(stream, end_point_copy[stage_point], cudaEventWaitDefault));

                    if constexpr (Config::debug) {
                        cudaEventRecord(start, stream);
                    }

                    // Do bucket sum
                    block_size = 256;
                    grid_size = num_sm;

                    bucket_sum<Config, 8, Point, PointAffine><<<grid_size, block_size, 0, stream>>>(
                        cur_len * Config::actual_windows,
                        cnt_zero,
                        indexs,
                        points[stage_point],
                        mutex,
                        initialized[j],
                        buckets_sum[j]
                    );
                    PROPAGATE_CUDA_ERROR(cudaGetLastError());

                    if constexpr (Config::debug) {
                        cudaEventRecord(stop, stream);
                        PROPAGATE_CUDA_ERROR(cudaEventSynchronize(stop));
                        cudaEventElapsedTime(&elapsedTime, start, stop);
                        std::cout << "MSM bucket sum time:" << elapsedTime << std::endl;
                    }

                    if (j == batches - 1) PROPAGATE_CUDA_ERROR(cudaEventRecord(begin_point_copy[stage_point], stream));

                    stage_scaler = (stage_scaler + 1) % max_scaler_stages;
                }

                if (p + stride != end) {
                    stage_point = (stage_point + 1) % max_point_stages;
                }
            }

            PROPAGATE_CUDA_ERROR(cudaStreamDestroy(copy_stream));

            cudaEvent_t start, stop;
            float ms;

            if constexpr (Config::debug) {
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
            }

            u32 grid = reduce_blocks; 

            // start reduce
            for (int j = 0; j < batches; j++) {
                if constexpr (Config::debug) {
                    cudaEventRecord(start, stream);
                }
                
                reduceBuckets<Config, 8> <<< grid, 256, 0, stream >>> (buckets_sum[j], reduce_buffer[j], initialized[j]);

                PROPAGATE_CUDA_ERROR(cudaGetLastError());

                if constexpr (Config::debug) {
                    cudaEventRecord(stop, stream);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&ms, start, stop);
                    std::cout << "MSM bucket reduce time:" << ms << std::endl;
                }
            }

            return cudaSuccess;
        }
 
        public:

        MSM(u64 len, u32 batch_per_run, u32 parts, u32 scaler_stages, u32 point_stages, int device = 0)
        : len(len), batch_per_run(batch_per_run), parts(parts), max_scaler_stages(scaler_stages), max_point_stages(point_stages),
        stage_scaler(0), stage_point(0), stage_point_transporting(0), device(device) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, device);
            num_sm = deviceProp.multiProcessorCount;
            reduce_blocks = div_ceil(num_sm, Config::n_windows) * Config::n_windows;
            d_buckets_sum_buf = new Point*[batch_per_run];
            buckets_sum = new Array2D<Point, Config::n_windows, Config::n_buckets>[batch_per_run];
            initialized_buf = new unsigned short*[batch_per_run];
            initialized = new Array2D<unsigned short, Config::n_windows, Config::n_buckets>[batch_per_run];
            scalers = new u32*[scaler_stages];
            points = new u32*[point_stages];
            begin_scaler_copy = new cudaEvent_t[scaler_stages];
            end_scaler_copy = new cudaEvent_t[scaler_stages];
            begin_point_copy = new cudaEvent_t[point_stages];
            end_point_copy = new cudaEvent_t[point_stages];
            reduce_buffer = new Point*[batch_per_run];
            h_reduce_buffer = new Point*[batch_per_run];
            
            for (int j = 0; j < batch_per_run; j++) {
                cudaMallocHost(&h_reduce_buffer[j], sizeof(Point) * reduce_blocks, cudaHostAllocDefault);
            }
            h_points_offset[0] = 0;
            for (u32 i = 0; i < Config::n_windows; i++) {
                h_points_per_window[i] = div_ceil(Config::actual_windows - i, Config::n_windows);
                h_points_offset[i + 1] = h_points_offset[i] + h_points_per_window[i];
            }
            // points_offset[Config::n_windows] should be the total number of points
            assert(h_points_offset[Config::n_windows] == Config::actual_windows);
        }

        void set_points(std::array<u32*, Config::n_precompute> host_points) {
            for (u32 i = 0; i < Config::n_precompute; i++) {
                h_points[i] = host_points[i];
            }
            points_set = true;
        }

        cudaError_t alloc_gpu(cudaStream_t stream = 0) {
            cudaError_t err;
            PROPAGATE_CUDA_ERROR(cudaSetDevice(device));
            u64 part_len = div_ceil(len, parts);
            for (u32 i = 0; i < batch_per_run; i++) {
                PROPAGATE_CUDA_ERROR(cudaMallocAsync(&d_buckets_sum_buf[i], sizeof(Point) * Config::n_windows * Config::n_buckets, stream));
                buckets_sum[i] = Array2D<Point, Config::n_windows, Config::n_buckets>(d_buckets_sum_buf[i]);
                PROPAGATE_CUDA_ERROR(cudaMallocAsync(&initialized_buf[i], sizeof(unsigned short) * (Config::n_buckets) * Config::n_windows, stream));
                initialized[i] = Array2D<unsigned short, Config::n_windows, Config::n_buckets>(initialized_buf[i]);
            }
            PROPAGATE_CUDA_ERROR(cudaMallocAsync(&mutex_buf, sizeof(unsigned short) * (Config::n_buckets) * Config::n_windows, stream));
            PROPAGATE_CUDA_ERROR(cudaMemsetAsync(mutex_buf, 0, sizeof(unsigned short) * (Config::n_buckets) * Config::n_windows, stream));
            PROPAGATE_CUDA_ERROR(cudaMallocAsync(&d_points_offset, sizeof(u32) * (Config::n_windows + 1), stream));
            PROPAGATE_CUDA_ERROR(cudaMemcpyAsync(d_points_offset, h_points_offset, sizeof(u32) * (Config::n_windows + 1), cudaMemcpyHostToDevice, stream));
            PROPAGATE_CUDA_ERROR(cudaMallocAsync(&cnt_zero, sizeof(u32), stream));
            PROPAGATE_CUDA_ERROR(cudaMallocAsync(&indexs, sizeof(u64) * Config::actual_windows * part_len * 2, stream));
            for (int i = 0; i < max_scaler_stages; i++) {
                PROPAGATE_CUDA_ERROR(cudaMallocAsync(&scalers[i], sizeof(u32) * Number::LIMBS * part_len, stream));
            }
            for (int i = 0; i < max_point_stages; i++) {
                PROPAGATE_CUDA_ERROR(cudaMallocAsync(&points[i], sizeof(u32) * PointAffine::N_WORDS * part_len * Config::n_precompute, stream));
            }
            cub::DeviceRadixSort::SortKeys(
                nullptr, temp_storage_bytes_sort,
                indexs + part_len * Config::actual_windows, indexs,
                Config::actual_windows * part_len, 0, Config::s, stream
            );
            PROPAGATE_CUDA_ERROR(cudaMallocAsync(&d_temp_storage_sort, temp_storage_bytes_sort, stream));
            for (int i = 0; i < batch_per_run; i++) {
                PROPAGATE_CUDA_ERROR(cudaMallocAsync(&reduce_buffer[i], sizeof(Point) * reduce_blocks, stream));
            }
            for (int i = 0; i < max_scaler_stages; i++) {
                PROPAGATE_CUDA_ERROR(cudaEventCreate(begin_scaler_copy + i));
                PROPAGATE_CUDA_ERROR(cudaEventCreate(end_scaler_copy + i));
            }
            for (int i = 0; i < max_point_stages; i++) {
                PROPAGATE_CUDA_ERROR(cudaEventCreate(begin_point_copy + i));
                PROPAGATE_CUDA_ERROR(cudaEventCreate(end_point_copy + i));
            }
            allocated = true;
            return cudaSuccess;
        }

        cudaError_t free_gpu(cudaStream_t stream = 0) {
            cudaError_t err;
            PROPAGATE_CUDA_ERROR(cudaSetDevice(device));
            if (!allocated) return cudaSuccess;
            for (u32 i = 0; i < batch_per_run; i++) {
                PROPAGATE_CUDA_ERROR(cudaFreeAsync(d_buckets_sum_buf[i], stream));
                PROPAGATE_CUDA_ERROR(cudaFreeAsync(initialized_buf[i], stream));
            }
            PROPAGATE_CUDA_ERROR(cudaFreeAsync(mutex_buf, stream));
            PROPAGATE_CUDA_ERROR(cudaFreeAsync(d_points_offset, stream));
            PROPAGATE_CUDA_ERROR(cudaFreeAsync(cnt_zero, stream));
            PROPAGATE_CUDA_ERROR(cudaFreeAsync(indexs, stream));
            for (int i = 0; i < max_scaler_stages; i++) {
                PROPAGATE_CUDA_ERROR(cudaFreeAsync(scalers[i], stream));
            }
            for (int i = 0; i < max_point_stages; i++) {
                PROPAGATE_CUDA_ERROR(cudaFreeAsync(points[i], stream));
            }
            PROPAGATE_CUDA_ERROR(cudaFreeAsync(d_temp_storage_sort, stream));
            for (int i = 0; i < batch_per_run; i++) {
                PROPAGATE_CUDA_ERROR(cudaFreeAsync(reduce_buffer[i], stream));
            }
            for (int i = 0; i < max_scaler_stages; i++) {
                PROPAGATE_CUDA_ERROR(cudaEventDestroy(begin_scaler_copy[i]));
                PROPAGATE_CUDA_ERROR(cudaEventDestroy(end_scaler_copy[i]));
            }
            for (int i = 0; i < max_point_stages; i++) {
                PROPAGATE_CUDA_ERROR(cudaEventDestroy(begin_point_copy[i]));
                PROPAGATE_CUDA_ERROR(cudaEventDestroy(end_point_copy[i]));
            }
            allocated = false;
            return cudaSuccess;
        }

        cudaError_t msm(const std::vector<u32*>& h_scalers, std::vector<Point> &h_result, cudaStream_t stream = 0) {
            // note: this function is not async, it will block until all computation is done
            // this is because the host reduce is done on the cpu
            assert(points_set);
            assert(h_scalers.size() == h_result.size());

            const u32 batches = h_scalers.size();
            cudaError_t err;
            PROPAGATE_CUDA_ERROR(cudaSetDevice(device));
            if (!allocated) PROPAGATE_CUDA_ERROR(alloc_gpu(stream));
            // TODO: use thread pool
            std::thread host_reduce_thread; // overlap host reduce with GPU computation

            auto host_reduce = [](Point **reduce_buffer, std::vector<Point>::iterator h_result, u32 n_reduce, u32 batches, cudaEvent_t start_reduce) {
                cudaEventSynchronize(start_reduce);
                // host timer
                std::chrono::high_resolution_clock::time_point start, end;
                if constexpr (Config::debug) {
                    start = std::chrono::high_resolution_clock::now();
                }
                for (u32 j = 0; j < batches; j++, h_result++) {
                    *h_result = Point::identity();
                    for (u32 i = 0; i < n_reduce; i++) {
                        *h_result = *h_result + reduce_buffer[j][i];
                    }
                }
                if constexpr (Config::debug) {
                    end = std::chrono::high_resolution_clock::now();
                    std::cout << "MSM host reduce time:" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << "ms" << std::endl;
                }
            };

            for (u32 i = 0; i < batches; i += batch_per_run) {
                u32 cur_batch = std::min(batch_per_run, batches - i);
                cudaEvent_t start_reduce;
                PROPAGATE_CUDA_ERROR(cudaEventCreateWithFlags(&start_reduce, cudaEventBlockingSync));
                PROPAGATE_CUDA_ERROR(run(cur_batch, h_scalers.begin() + i, i == 0, stream));

                if (i > 0) host_reduce_thread.join();
                
                for (int j = 0; j < cur_batch; j++) {
                    PROPAGATE_CUDA_ERROR(cudaMemcpyAsync(h_reduce_buffer[j], reduce_buffer[j], sizeof(Point) * reduce_blocks, cudaMemcpyDeviceToHost, stream));
                }
                PROPAGATE_CUDA_ERROR(cudaEventRecord(start_reduce, stream));

                host_reduce_thread = std::thread(host_reduce, h_reduce_buffer, h_result.begin() + i, reduce_blocks, cur_batch, start_reduce);
            }

            host_reduce_thread.join();

            PROPAGATE_CUDA_ERROR(cudaGetLastError());

            return cudaSuccess;
        }

        ~MSM() {
            free_gpu();
            delete [] d_buckets_sum_buf;
            delete [] buckets_sum;
            delete [] initialized_buf;
            delete [] initialized;
            delete [] scalers;
            delete [] points;
            delete [] begin_scaler_copy;
            delete [] end_scaler_copy;
            delete [] begin_point_copy;
            delete [] end_point_copy;
            delete [] reduce_buffer;
            for (int j = 0; j < batch_per_run; j++) {
                cudaFreeHost(h_reduce_buffer[j]);
            }
            delete [] h_reduce_buffer;
        }
    };

    template <typename Config, typename Point = bn254::Point, typename PointAffine = bn254::PointAffine>
    class MSMPrecompute {
        static cudaError_t run(u64 len, u64 part_len, std::array<u32*, Config::n_precompute> h_points, cudaStream_t stream = 0) {
            cudaError_t err;
            if constexpr (Config::n_precompute == 1) {
                return cudaSuccess;
            }
            u32 *points;
            PROPAGATE_CUDA_ERROR(cudaMallocAsync(&points, sizeof(PointAffine) * part_len * Config::n_precompute, stream));
            for (int i = 0; i * part_len < len; i++) {
                u64 offset = i * part_len;
                u64 cur_len = std::min(part_len, len - offset);

                PROPAGATE_CUDA_ERROR(cudaMemcpyAsync(points, h_points[0] + offset * PointAffine::N_WORDS, sizeof(PointAffine) * part_len, cudaMemcpyHostToDevice, stream));

                u32 grid = div_ceil(cur_len, 256);
                u32 block = 256;
                precompute_kernel<Config, Point, PointAffine><<<grid, block, 0, stream>>>(points, cur_len);

                for (int j = 1; j < Config::n_precompute; j++) {
                    PROPAGATE_CUDA_ERROR(cudaMemcpyAsync(h_points[j] + offset * PointAffine::N_WORDS, points + j * cur_len * PointAffine::N_WORDS, sizeof(PointAffine) * cur_len, cudaMemcpyDeviceToHost, stream));
                }
            }
            PROPAGATE_CUDA_ERROR(cudaFreeAsync(points, stream));
            return cudaSuccess;
        }

        public:
        // synchronous function
        static cudaError_t precompute(u64 len, std::array<u32*, Config::n_precompute> h_points, int max_devices = 1) {
            cudaError_t err;
            int devices;
            PROPAGATE_CUDA_ERROR(cudaGetDeviceCount(&devices));
            devices = std::min(devices, max_devices);
            if (Config::debug) std::cout << "Using " << devices << " devices" << std::endl;
            std::vector<cudaStream_t> streams;
            streams.resize(devices);
            u64 part_len = div_ceil(len, devices);
            for (u32 i = 0; i < devices; i++) {
                if (Config::debug) std::cout << "Precomputing on device " << i << std::endl;
                u64 offset = i * part_len;
                u64 cur_len = std::min(part_len, len - offset);
                PROPAGATE_CUDA_ERROR(cudaSetDevice(i));
                PROPAGATE_CUDA_ERROR(cudaStreamCreate(&streams[i]));
                std::array<u32*, Config::n_precompute> cur_points;
                for (int j = 0; j < Config::n_precompute; j++) {
                    cur_points[j] = h_points[j] + offset * PointAffine::N_WORDS;
                }
                PROPAGATE_CUDA_ERROR(run(cur_len, std::min(cur_len, 1ul << 20), cur_points, streams[i]));
            }
            for (u32 i = 0; i < devices; i++) {
                PROPAGATE_CUDA_ERROR(cudaStreamSynchronize(streams[i]));
                PROPAGATE_CUDA_ERROR(cudaStreamDestroy(streams[i]));
            }
            return cudaSuccess;
        }
    };

    // each msm will be decomposed into multiple msm instances, each instance will be run on a single GPU
    template <typename Config, typename Number = bn254_fr::Number, typename Point = bn254::Point, typename PointAffine = bn254::PointAffine>
    class MultiGPUMSM {
        u32 parts, scaler_stages, point_stages, batch_per_run;
        u64 len, part_len;
        const std::vector<u32> cards;
        std::vector<MSM<Config, Number, Point, PointAffine>> msm_instances;
        std::vector<cudaStream_t> streams;
        // TODO: use thread pool
        std::vector<std::thread> threads;
        bool allocated = false;
        public:
        MultiGPUMSM(u64 len, u32 batch_per_run, u32 parts, u32 scaler_stages, u32 point_stages, std::vector<u32> cards)
        : len(len), part_len(div_ceil(len, cards.size())), batch_per_run(batch_per_run), parts(parts),
        scaler_stages(scaler_stages), point_stages(point_stages), cards(cards) {
            msm_instances.reserve(cards.size());
            streams.resize(cards.size());
            threads.resize(cards.size());
            for (u32 i = 0; i < cards.size(); i++) {
                u64 offset = i * part_len;
                u64 cur_len = std::min(part_len, len - offset);
                msm_instances.emplace_back(cur_len, batch_per_run, parts, scaler_stages, point_stages, cards[i]);
            }
        }

        cudaError_t alloc_gpu() {
            cudaError_t err;
            if (allocated) return cudaSuccess;
            for (u32 i = 0; i < cards.size(); i++) {
                PROPAGATE_CUDA_ERROR(cudaSetDevice(cards[i]));
                PROPAGATE_CUDA_ERROR(msm_instances[i].alloc_gpu());
                PROPAGATE_CUDA_ERROR(cudaStreamCreate(&streams[i]));
            }
            allocated = true;
            return cudaSuccess;
        }

        cudaError_t free_gpu() {
            cudaError_t err;
            if (!allocated) return cudaSuccess;
            for (u32 i = 0; i < cards.size(); i++) {
                PROPAGATE_CUDA_ERROR(cudaSetDevice(cards[i]));
                PROPAGATE_CUDA_ERROR(msm_instances[i].free_gpu(streams[i]));
                PROPAGATE_CUDA_ERROR(cudaStreamSynchronize(streams[i]));
                PROPAGATE_CUDA_ERROR(cudaStreamDestroy(streams[i]));
            }
            allocated = false;
            return cudaSuccess;
        }

        void set_points(std::array<u32*, Config::n_precompute> host_points) {
            for (u32 i = 0; i < cards.size(); i++) {
                u64 offset = i * part_len;
                u64 cur_len = std::min(part_len, len - offset);
                std::array<u32*, Config::n_precompute> cur_points;
                for (int j = 0; j < Config::n_precompute; j++) {
                    cur_points[j] = host_points[j] + offset * PointAffine::N_WORDS;
                }
                msm_instances[i].set_points(cur_points);
            }
        }

        cudaError_t msm(const std::vector<u32*>& h_scalers, std::vector<Point> &h_result) {
            assert(h_scalers.size() == h_result.size());

            cudaError_t err;
            if (!allocated) PROPAGATE_CUDA_ERROR(alloc_gpu());

            // pre-card results
            std::vector<std::vector<Point>> results(cards.size());
            for (u32 i = 0; i < cards.size(); i++) {
                results[i].resize(h_result.size());
            }
            
            auto run_msm = [](MSM<Config, Number, Point, PointAffine> &msm, const std::vector<u32*> &h_scalers, std::vector<Point> &h_result, cudaStream_t stream) {
                msm.msm(h_scalers, h_result, stream);
            };

            for (u32 i = 0; i < cards.size(); i++) {
                u64 offset = i * part_len;
                u64 cur_len = std::min(part_len, len - offset);
                std::vector<u32*> cur_scalers = h_scalers;
                for (u32 j = 0; j < h_scalers.size(); j++) {
                    cur_scalers[j] += offset * Number::LIMBS;
                }
                threads[i] = std::thread(run_msm, std::ref(msm_instances[i]), cur_scalers, std::ref(results[i]), streams[i]);
            }

            for (u32 i = 0; i < cards.size(); i++) {
                threads[i].join();
            }

            for (u32 i = 0; i < h_result.size(); i++) {
                h_result[i] = Point::identity();
                for (u32 j = 0; j < cards.size(); j++) {
                    h_result[i] = h_result[i] + results[j][i];
                }
            }
            return cudaSuccess;
        }

        ~MultiGPUMSM() {
            free_gpu();
        }
    };
}