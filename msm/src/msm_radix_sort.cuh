#include "../../mont/src/bn254_scalar.cuh"
#include "bn254.cuh"
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

#include <cub/cub.cuh>
#include <iostream>

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

    using bn254::Point;
    using bn254::PointAffine;
    using bn254_scalar::Element;
    using bn254_scalar::Number;
    using mont::u32;
    using mont::u64;

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
    u32 TARGET_WINDOWS = 1, u32 SEGMENTS = 1,
    u32 SCALER_STAGES = 2, u32 POINT_STAGES = 2, bool DEBUG = true>
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

        // number of parts to divide the input
        static constexpr u32 n_parts = SEGMENTS;

        static constexpr bool debug = DEBUG;

        static constexpr u32 scaler_stages = std::min(SCALER_STAGES, n_parts);

        static constexpr u32 point_stages = std::min(POINT_STAGES, n_parts);
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
    template <typename Config>
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

    template <typename Config, u32 WarpPerBlock>
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
                unsigned short *mutex_ptr;
                if unlikely(first) {
                    mutex_ptr = mutex.addr(window_id, key - 1);
                    lock(mutex_ptr);
                }

                if (initialized.get(window_id, key - 1)) {
                    sum.get(window_id, key - 1) = sum.get(window_id, key - 1) + acc;
                } else {
                    sum.get(window_id, key - 1) = acc;
                    initialized.get(window_id, key - 1) = 1;
                }
                if unlikely(first) unlock(mutex_ptr);
                first = false;
                acc = Point::identity();
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

    template<typename Config, u32 WarpNum = 8>
    __launch_bounds__(256,1)
    __global__ void reduceBuckets(
        Array2D<Point, Config::n_windows, Config::n_buckets> buckets_sum, 
        Point *reduceMemory,
        Array2D<unsigned short, Config::n_windows, Config::n_buckets> initialized
    ) {

        assert(gridDim.x % Config::n_windows == 0);

        __shared__ u32 smem[WarpNum][Point::N_WORDS + 4]; // +4 for padding

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

        if (threadIdx.x < WarpNum) {
            sum_of_sums = Point::load(smem[threadIdx.x]);
        } else {
            sum_of_sums = Point::identity();
        }

        // Reduce in warp1
        if constexpr (WarpNum > 16) sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(16);
        if constexpr (WarpNum > 8) sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(8);
        if constexpr (WarpNum > 4) sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(4);
        if constexpr (WarpNum > 2) sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(2);
        if constexpr (WarpNum > 1) sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(1);

        // Store to global memory
        if (threadIdx.x == 0) {
            for (u32 i = 0; i < window_id * Config::s; i++) {
               sum_of_sums = sum_of_sums.self_add();
            }
            reduceMemory[blockIdx.x] = sum_of_sums;
        }
    }

    template <typename Config>
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

    template <typename Config>
    cudaError_t precompute(
        const u32 *h_points,
        u64 len,
        u32 *&h_points_precompute,
        u32 &head, // d_points store head or tail
        cudaStream_t stream = 0
    ) {
        cudaError_t err;
        u64 part_len = div_ceil(len, Config::n_parts);
        head = 1;
        PROPAGATE_CUDA_ERROR(cudaHostAlloc(&h_points_precompute, sizeof(PointAffine) * len * Config::n_precompute, cudaHostAllocDefault));

        if constexpr (Config::n_precompute == 1) {
            memcpy(h_points_precompute, h_points, sizeof(PointAffine) * len);
            return cudaSuccess;
        }

        cudaEvent_t begin_precompute;
        PROPAGATE_CUDA_ERROR(cudaEventCreate(&begin_precompute));
        PROPAGATE_CUDA_ERROR(cudaEventRecord(begin_precompute, stream));
        cudaStream_t child_stream[2];
        PROPAGATE_CUDA_ERROR(cudaStreamCreate(&child_stream[0]));
        if constexpr (Config::n_parts > 1) PROPAGATE_CUDA_ERROR(cudaStreamCreate(&child_stream[1]));
        PROPAGATE_CUDA_ERROR(cudaStreamWaitEvent(child_stream[0], begin_precompute, cudaEventWaitDefault));
        if constexpr (Config::n_parts > 1) PROPAGATE_CUDA_ERROR(cudaStreamWaitEvent(child_stream[1], begin_precompute, cudaEventWaitDefault));

        u32 *points[2];
        PROPAGATE_CUDA_ERROR(cudaMallocAsync(&points[0], sizeof(PointAffine) * part_len * Config::n_precompute, child_stream[0]));
        if constexpr (Config::n_parts > 1) PROPAGATE_CUDA_ERROR(cudaMallocAsync(&points[1], sizeof(PointAffine) * part_len * Config::n_precompute, child_stream[1]));

        u32 stage = 0;

        for (int i = Config::n_parts - 1; i >= 0; i--) {
            u64 offset = i * part_len;
            u64 cur_len = std::min(part_len, len - offset);

            PROPAGATE_CUDA_ERROR(cudaMemcpyAsync(points[stage], h_points + offset * PointAffine::N_WORDS, sizeof(PointAffine) * part_len, cudaMemcpyHostToDevice, child_stream[stage]));

            u32 grid = div_ceil(cur_len, 256);
            u32 block = 256;
            precompute_kernel<Config><<<grid, block, 0, child_stream[stage]>>>(points[stage], cur_len);

            PROPAGATE_CUDA_ERROR(cudaMemcpyAsync(h_points_precompute + offset * Config::n_precompute * PointAffine::N_WORDS, points[stage], sizeof(PointAffine) * part_len * Config::n_precompute, cudaMemcpyDeviceToHost, child_stream[stage]));
            stage ^= 1;
        }
        PROPAGATE_CUDA_ERROR(cudaFreeAsync(points[stage^1], child_stream[stage^1]));
        if constexpr (Config::n_parts > 1) PROPAGATE_CUDA_ERROR(cudaFreeAsync(points[stage], child_stream[stage]));
        cudaEvent_t end_precompute[2];
        PROPAGATE_CUDA_ERROR(cudaEventCreate(&end_precompute[0]));
        PROPAGATE_CUDA_ERROR(cudaEventRecord(end_precompute[0], child_stream[0]));
        if constexpr (Config::n_parts > 1) {
            PROPAGATE_CUDA_ERROR(cudaEventCreate(&end_precompute[1]));
            PROPAGATE_CUDA_ERROR(cudaEventRecord(end_precompute[1], child_stream[1]));
        }
        PROPAGATE_CUDA_ERROR(cudaStreamWaitEvent(stream, end_precompute[0], cudaEventWaitDefault));
        if constexpr (Config::n_parts > 1) PROPAGATE_CUDA_ERROR(cudaStreamWaitEvent(stream, end_precompute[1], cudaEventWaitDefault));

        return cudaSuccess;
    }


    template <typename Config>
    __host__ cudaError_t run(
        const u64 len,
        const u32 *h_scalers,
        const u32 *h_points_precompute,
        Point &h_result,
        bool on_gpu,
        bool keep_points,
        u32 *&d_points,
        u32 &head,
        cudaStream_t stream = 0
    ) {
        cudaError_t err;

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

        Point *buckets_sum_buf;
        PROPAGATE_CUDA_ERROR(cudaMallocAsync(&buckets_sum_buf, sizeof(Point) * Config::n_windows * (Config::n_buckets), stream));
        auto buckets_sum = Array2D<Point, Config::n_windows, Config::n_buckets>(buckets_sum_buf);
        
        unsigned short *mutex_buf;
        PROPAGATE_CUDA_ERROR(cudaMallocAsync(&mutex_buf, sizeof(unsigned short) * (Config::n_buckets) * Config::n_windows, stream));
        PROPAGATE_CUDA_ERROR(cudaMemsetAsync(mutex_buf, 0, sizeof(unsigned short) * (Config::n_buckets) * Config::n_windows, stream));
        Array2D<unsigned short, Config::n_windows, Config::n_buckets> mutex(mutex_buf);

        unsigned short *initialized_buf;
        PROPAGATE_CUDA_ERROR(cudaMallocAsync(&initialized_buf, sizeof(unsigned short) * (Config::n_buckets) * Config::n_windows, stream));
        PROPAGATE_CUDA_ERROR(cudaMemsetAsync(initialized_buf, 0, sizeof(unsigned short) * (Config::n_buckets) * Config::n_windows, stream));
        Array2D<unsigned short, Config::n_windows, Config::n_buckets> initialized(initialized_buf);

        // record for number of logical windows in each actual window
        u32 h_points_per_window[Config::n_windows];
        u32 h_points_offset[Config::n_windows + 1];
        h_points_offset[0] = 0;
        for (u32 i = 0; i < Config::n_windows; i++) {
            h_points_per_window[i] = div_ceil(Config::actual_windows - i, Config::n_windows);
            h_points_offset[i + 1] = h_points_offset[i] + h_points_per_window[i];
        }
        
        // points_offset[Config::n_windows] should be the total number of points
        assert(h_points_offset[Config::n_windows] == Config::actual_windows);

        // read only, no need to double buffer
        u32 *d_points_offset;
        PROPAGATE_CUDA_ERROR(cudaMallocAsync(&d_points_offset, sizeof(u32) * (Config::n_windows + 1), stream));
        PROPAGATE_CUDA_ERROR(cudaMemcpyAsync(d_points_offset, h_points_offset, sizeof(u32) * (Config::n_windows + 1), cudaMemcpyHostToDevice, stream));

        u64 part_len = div_ceil(len, Config::n_parts);

        u32 *cnt_zero;
        PROPAGATE_CUDA_ERROR(cudaMallocAsync(&cnt_zero, sizeof(u32), stream));

        // indexs is used to store the bucket id, point index and sign
        // Config::s bits for bucket id, 1 bit for sign, Config::window_bits for window id, the rest for point index
        // max log(2^30(max points) * precompute) + Config::s bits are needed
        static_assert(log2_ceil(Config::n_precompute) + 30 + Config::window_bits + Config::s + 1 <= 64, "Index too large");
        // for sorting, after sort, points with same bucket id are gathered, gives pointer to original index
        u64 *indexs;
        PROPAGATE_CUDA_ERROR(cudaMallocAsync(&indexs, sizeof(u64) * Config::actual_windows * part_len * 2, stream));
        u32 *scalers[Config::scaler_stages];
        for (int i = 0; i < Config::scaler_stages; i++) {
            PROPAGATE_CUDA_ERROR(cudaMallocAsync(&scalers[i], sizeof(u32) * Number::LIMBS * part_len, stream));
        }

        void *d_temp_storage;
        size_t temp_storage_bytes = 0;

        cub::DeviceRadixSort::SortKeys(
            nullptr, temp_storage_bytes,
            indexs + part_len * Config::actual_windows, indexs,
            Config::actual_windows * part_len, 0, Config::s, stream
        );

        PROPAGATE_CUDA_ERROR(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));

        u32* points[Config::point_stages];
        if (on_gpu) points[0] = d_points;
        else PROPAGATE_CUDA_ERROR(cudaMallocAsync(&points[0], sizeof(PointAffine) * part_len * Config::n_precompute, stream));
        for (int i = 1; i < Config::point_stages; i++) {
            PROPAGATE_CUDA_ERROR(cudaMallocAsync(&points[i], sizeof(PointAffine) * part_len * Config::n_precompute, stream));
        }

        int begin, end, stride;
        if (!on_gpu) {
            begin = 0;
            end = Config::n_parts;
            stride = 1;
            if (keep_points) head = 0;
        } else {
            if (head) {
                begin = 0;
                end = Config::n_parts;
                stride = 1;
            } else {
                begin = Config::n_parts - 1;
                end = -1;
                stride = -1;
            }
            head ^= 1;
        }
        
        cudaStream_t copy_stream;
        PROPAGATE_CUDA_ERROR(cudaStreamCreate(&copy_stream));

        cudaEvent_t begin_scaler_copy[Config::scaler_stages], end_scaler_copy[Config::scaler_stages];
        cudaEvent_t begin_point_copy[Config::point_stages], end_point_copy[Config::point_stages];

        for (int i = 0; i < Config::scaler_stages; i++) {
            PROPAGATE_CUDA_ERROR(cudaEventCreate(&begin_scaler_copy[i]));
            PROPAGATE_CUDA_ERROR(cudaEventCreate(&end_scaler_copy[i]));
            PROPAGATE_CUDA_ERROR(cudaEventRecord(begin_scaler_copy[i], stream));
        }
        for (int i = 0; i < Config::point_stages; i++) {
            PROPAGATE_CUDA_ERROR(cudaEventCreate(&begin_point_copy[i]));
            PROPAGATE_CUDA_ERROR(cudaEventCreate(&end_point_copy[i]));
            PROPAGATE_CUDA_ERROR(cudaEventRecord(begin_point_copy[i], stream));
        }

        int stage_scaler = 0, stage_point = 0;

        for (int p = begin; p != end; p += stride) {
            u64 offset = p * part_len;
            u64 cur_len = std::min(part_len, len - offset);

            PROPAGATE_CUDA_ERROR(cudaStreamWaitEvent(copy_stream, begin_scaler_copy[stage_scaler], cudaEventWaitDefault));
            PROPAGATE_CUDA_ERROR(cudaMemcpyAsync(scalers[stage_scaler], h_scalers + offset * Number::LIMBS, sizeof(u32) * Number::LIMBS * cur_len, cudaMemcpyHostToDevice, copy_stream));
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
            u32 grid_size = deviceProp.multiProcessorCount;
            distribute_windows<Config><<<grid_size, block_size, 0, stream>>>(
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
                d_temp_storage, temp_storage_bytes,
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

            PROPAGATE_CUDA_ERROR(cudaStreamWaitEvent(copy_stream, begin_point_copy[stage_point], cudaEventWaitDefault));
            if (!on_gpu || p != begin) {
                PROPAGATE_CUDA_ERROR(cudaMemcpyAsync(points[stage_point], h_points_precompute + offset * Config::n_precompute * PointAffine::N_WORDS, sizeof(PointAffine) * cur_len * Config::n_precompute, cudaMemcpyHostToDevice, copy_stream));
            }
            PROPAGATE_CUDA_ERROR(cudaEventRecord(end_point_copy[stage_point], copy_stream));

            PROPAGATE_CUDA_ERROR(cudaStreamWaitEvent(stream, end_point_copy[stage_point], cudaEventWaitDefault));

            if constexpr (Config::debug) {
                cudaEventRecord(start, stream);
            }

            // Do bucket sum
            block_size = 256;
            grid_size = deviceProp.multiProcessorCount;

            bucket_sum<Config, 8><<<grid_size, block_size, 0, stream>>>(
                cur_len * Config::actual_windows,
                cnt_zero,
                indexs,
                points[stage_point],
                mutex,
                initialized,
                buckets_sum
            );
            PROPAGATE_CUDA_ERROR(cudaGetLastError());

            if constexpr (Config::debug) {
                cudaEventRecord(stop, stream);
                PROPAGATE_CUDA_ERROR(cudaEventSynchronize(stop));
                cudaEventElapsedTime(&elapsedTime, start, stop);
                std::cout << "MSM bucket sum time:" << elapsedTime << std::endl;
            }

            PROPAGATE_CUDA_ERROR(cudaEventRecord(begin_point_copy[stage_point], stream));
            if (p + stride != end) {
                stage_scaler = (stage_scaler + 1) % Config::scaler_stages;
                stage_point = (stage_point + 1) % Config::point_stages;
            }
        }

        if (keep_points) d_points = points[stage_point];
        else d_points = nullptr;

        for (int i = 0; i < Config::scaler_stages; i++) {
            PROPAGATE_CUDA_ERROR(cudaFreeAsync(scalers[i], stream));
        }

        PROPAGATE_CUDA_ERROR(cudaFreeAsync(indexs, stream));
        PROPAGATE_CUDA_ERROR(cudaFreeAsync(d_temp_storage, stream));
        PROPAGATE_CUDA_ERROR(cudaFreeAsync(cnt_zero, stream));

        for (int i = 0; i < Config::point_stages; i++) {
            if ((!keep_points) || i != stage_point) {
                PROPAGATE_CUDA_ERROR(cudaFreeAsync(points[i], stream));
            }
        }
        
        PROPAGATE_CUDA_ERROR(cudaFreeAsync(mutex_buf, stream));
        PROPAGATE_CUDA_ERROR(cudaFreeAsync(d_points_offset, stream));
        for (int i = 0; i < Config::scaler_stages; i++) {
            PROPAGATE_CUDA_ERROR(cudaEventDestroy(begin_scaler_copy[i]));
            PROPAGATE_CUDA_ERROR(cudaEventDestroy(end_scaler_copy[i]));
        }
        for (int i = 0; i < Config::point_stages; i++) {
            PROPAGATE_CUDA_ERROR(cudaEventDestroy(begin_point_copy[i]));
            PROPAGATE_CUDA_ERROR(cudaEventDestroy(end_point_copy[i]));
        }

        PROPAGATE_CUDA_ERROR(cudaStreamDestroy(copy_stream));

        cudaEvent_t start, stop;
        float ms;

        if constexpr (Config::debug) {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
        }

        u32 grid = div_ceil(deviceProp.multiProcessorCount, Config::n_windows) * Config::n_windows; 
        Point *reduce_buffer;
        PROPAGATE_CUDA_ERROR(cudaMallocAsync(&reduce_buffer, sizeof(Point) * grid, stream));

        // start reduce
        if constexpr (Config::debug) {
            cudaEventRecord(start, stream);
        }
        
        reduceBuckets<Config, 8> <<< grid, 256, 0, stream >>> (buckets_sum, reduce_buffer, initialized);

        PROPAGATE_CUDA_ERROR(cudaGetLastError());

        if constexpr (Config::debug) {
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            std::cout << "MSM bucket reduce time:" << ms << std::endl;
        }

        PROPAGATE_CUDA_ERROR(cudaFreeAsync(buckets_sum_buf, stream));
        PROPAGATE_CUDA_ERROR(cudaFreeAsync(initialized_buf, stream));

        // ruduce all host
        Point *h_reduce_buffer;
        PROPAGATE_CUDA_ERROR(cudaMallocHost(&h_reduce_buffer, sizeof(Point) * grid));
        PROPAGATE_CUDA_ERROR(cudaMemcpyAsync(h_reduce_buffer, reduce_buffer, sizeof(Point) * grid, cudaMemcpyDeviceToHost, stream));
        PROPAGATE_CUDA_ERROR(cudaFreeAsync(reduce_buffer, stream));
        PROPAGATE_CUDA_ERROR(cudaStreamSynchronize(stream));

        if constexpr (Config::debug) {
            cudaEventRecord(start, stream);
        }

        h_result = Point::identity();

        for (u32 i = 0; i < grid; i++) {
            h_result = h_result + h_reduce_buffer[i];
        }

        if constexpr (Config::debug) {
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            std::cout << "MSM cpu reduce time:" << ms << std::endl;
        }

        return cudaSuccess;
    }
}