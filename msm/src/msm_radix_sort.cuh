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

    template <u32 BITS = 256, u32 WINDOW_SIZE = 22, u32 TARGET_WINDOWS = 1, u32 SEGMENTS = 1>
    struct MsmConfig {
        static constexpr u32 lambda = BITS;
        static constexpr u32 s = WINDOW_SIZE;
        static constexpr u32 n_buckets = pow2(s);

        // # of logical windows
        static constexpr u32 actual_windows = div_ceil(lambda, s);
        
        // stride for precomputation(same as # of windows), if >= actual_windows, no precomputation; if = 1, precompute all
        static constexpr u32 n_windows = actual_windows < TARGET_WINDOWS ? actual_windows : TARGET_WINDOWS;
        
        // lines of points to be stored in memory, 1 for no precomputation
        static constexpr u32 n_precompute = div_ceil(actual_windows, n_windows);

        // number of parts to divide the input
        static constexpr u32 n_parts = SEGMENTS;

        static constexpr bool debug = true;
    };

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
            u32 bucket[Config::actual_windows];
            auto scaler = Number::load(scalers + i * Number::LIMBS);
            scaler.bit_slice<Config::actual_windows, Config::s>(bucket);

            #pragma unroll
            for (u32 window_id = 0; window_id < Config::actual_windows; window_id++) {
                auto bucket_id = bucket[window_id];
                u32 physical_window_id = window_id % Config::n_windows;
                u32 point_group = window_id / Config::n_windows;
                if (bucket_id == 0) {
                    atomicAdd(cnt_zero + physical_window_id, 1);
                    // printf("window_id: %d, cnt_zero: %d\n", physical_window_id, cnt_zero[physical_window_id]);
                }
                // if (i == 9) {
                //     printf("window_id: %d, bucket_id: %d, point_group: %d\n", physical_window_id, bucket_id, point_group);
                //     printf("points_offset[%d]: %d\n", physical_window_id, points_offset[physical_window_id]);
                //     printf("indexs[%lld]: %lld\n", (points_offset[physical_window_id] + point_group) * len + i, bucket_id | ((point_group * len + i) << Config::s));
                // }
                indexs[(points_offset[physical_window_id] + point_group) * len + i] = bucket_id | ((point_group * len + i) << Config::s);
            }
        }
    }

    __device__ __forceinline__ void lock(unsigned short *mutex_ptr, u32 wait_limit = 16) {
        u32 time = 1;
        while (atomicCAS(mutex_ptr, (unsigned short int)0, (unsigned short int)1) != 0) {
            __nanosleep(time);
            if (time < wait_limit) time *= 2;
        }
        __threadfence();
    }

    __device__ __forceinline__ void unlock(unsigned short *mutex_ptr) {
        __threadfence();
        atomicCAS(mutex_ptr, (unsigned short int)1, (unsigned short int)0);
    }

    template <typename Config, u32 WarpPerBlock>
    __global__ void bucket_sum(
        const u32 window_id,
        const u64 len,
        const u64 *indexs,
        const u32 *points,
        const u32 *cnt_zero,
        unsigned short *mutex,
        Array2D<unsigned short, Config::n_windows, Config::n_buckets - 1> initialized,
        Array2D<Point, Config::n_windows, Config::n_buckets - 1> sum
    ) {
        __shared__ u32 point_buffer[WarpPerBlock * THREADS_PER_WARP][PointAffine::N_WORDS + 4]; // +4 for padding and alignment

        const static u64 key_mask = (1u << Config::s) - 1;

        const u32 gtid = threadIdx.x + blockIdx.x * blockDim.x;
        const u32 threads = blockDim.x * gridDim.x;
        const u32 zero_offset = cnt_zero[window_id];
        indexs += zero_offset;

        u32 work_len = div_ceil(len - zero_offset, threads);
        if (work_len % 2 != 0) work_len++; // we need each thread to read two indexes at a time for 16 byte alignment

        const u32 start_id = work_len * gtid;
        const u32 end_id = min(start_id + work_len, (u32)len - zero_offset);
        if (start_id >= end_id) return;

        // for async transfer, 16 byte alignment is needed so we read two indexes at a time
        // if zero_offset is odd, means the first index is not aligned, so we need to start from the second index
        // when stage is 1, we need to read the next index for the next iteration
        int stage = zero_offset & 1u;

        PointAffine p;
        bool first = true; // only the first bucket and the last bucket may have conflict with other threads
        auto pip_thread = cuda::make_pipeline(); // pipeline for this thread

        u64 index = indexs[start_id];
        u64 pointer = index >> Config::s;
        u32 key = index & key_mask;

        pip_thread.producer_acquire();
        cuda::memcpy_async(reinterpret_cast<uint4*>(point_buffer[threadIdx.x]), reinterpret_cast<const uint4*>(points + pointer * PointAffine::N_WORDS), sizeof(PointAffine), pip_thread);
        if (stage == 0) cuda::memcpy_async(reinterpret_cast<uint4*>(point_buffer[threadIdx.x] + PointAffine::N_WORDS), reinterpret_cast<const uint4*>(indexs + start_id), sizeof(u64) * 2, pip_thread);
        else if (stage == 1) cuda::memcpy_async(reinterpret_cast<uint4*>(point_buffer[threadIdx.x] + PointAffine::N_WORDS), reinterpret_cast<const uint4*>(indexs + start_id + 1), sizeof(u64) * 2, pip_thread);
        stage ^= 1;
        pip_thread.producer_commit();

        auto acc = Point::identity();

        for (u32 i = start_id + 1; i < end_id; i++) {
            u64 next_index;
            // printf("pointer: %d\n", pointer);

            pip_thread.consumer_wait();
            p = PointAffine::load(point_buffer[threadIdx.x]);
            next_index = reinterpret_cast<const u64*>(point_buffer[threadIdx.x] + PointAffine::N_WORDS)[stage];
            pointer = next_index >> Config::s;
            pip_thread.consumer_release();

            pip_thread.producer_acquire();
            cuda::memcpy_async(reinterpret_cast<uint4*>(point_buffer[threadIdx.x]), reinterpret_cast<const uint4*>(points + pointer * PointAffine::N_WORDS), sizeof(PointAffine), pip_thread);
            if (stage == 1 && i + 1 < end_id) {
                cuda::memcpy_async(reinterpret_cast<uint4*>(point_buffer[threadIdx.x] + PointAffine::N_WORDS), reinterpret_cast<const uint4*>(indexs + i + 1), sizeof(u64) * 2, pip_thread);
            }
            stage ^= 1;
            pip_thread.producer_commit();

            // printf("key: %d\n", key);
            // Point tmp = Point::identity();
            // tmp = tmp + p;
            // for (u32 j = 0; j < window_id * Config::s; j++) {
            //    tmp = tmp.self_add();
            // }
            // auto pp = tmp.to_affine();
            // for (u32 j = 0; j < 8; j++) {
            //     printf("p[%d]: %d ", j, pp.x.n.limbs[j]);
            // }
            // printf("\n");

            acc = acc + p;

            u32 next_key = next_index & key_mask;

            if (next_key != key) {
                auto mutex_ptr = mutex + key;
                if (first) lock(mutex_ptr);

                if (initialized.get(window_id, key - 1)) {
                    sum.get(window_id, key - 1) = sum.get(window_id, key - 1) + acc;
                } else {
                    sum.get(window_id, key - 1) = acc;
                    initialized.get(window_id, key - 1) = 1;
                }

                if (first) {
                    first = false;                    
                    unlock(mutex_ptr);
                }
                acc = Point::identity();
            }
            key = next_key;
        }

        pip_thread.consumer_wait();
        p = PointAffine::load(point_buffer[threadIdx.x]);
        pip_thread.consumer_release();

        // printf("pointer: %d\n", pointer);
        // printf("key: %d\n", key);
        // Point tmp = Point::identity();
        // tmp = tmp + p;
        // for (u32 j = 0; j < window_id * Config::s; j++) {
        //     tmp = tmp.self_add();
        // }
        // auto pp = tmp.to_affine();
        // for (u32 j = 0; j < 8; j++) {
        //     printf("p[%d]: %d ", j, pp.x.n.limbs[j]);
        // }
        // printf("\n");

        acc = acc + p;

        auto mutex_ptr = mutex + key;
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
        Array2D<Point, Config::n_windows, Config::n_buckets - 1> buckets_sum, 
        Point *reduceMemory,
        Array2D<unsigned short, Config::n_windows, Config::n_buckets - 1> initialized
    ) {

        assert(gridDim.x % Config::n_windows == 0);

        __shared__ u32 smem[WarpNum][Point::N_WORDS + 4]; // +4 for padding and alignment

        const u32 total_threads_per_window = gridDim.x / Config::n_windows * blockDim.x;
        u32 window_id = blockIdx.x / (gridDim.x / Config::n_windows);

        u32 wtid = (blockIdx.x % (gridDim.x / Config::n_windows)) * blockDim.x + threadIdx.x;
          
        const u32 buckets_per_thread = div_ceil(Config::n_buckets - 1, total_threads_per_window);

        Point sum, sum_of_sums;

        sum = Point::identity();
        sum_of_sums = Point::identity();

        for(u32 i=buckets_per_thread; i > 0; i--) {
            u32 loadIndex = wtid * buckets_per_thread + i;
            if(loadIndex < Config::n_buckets && initialized.get(window_id, loadIndex - 1)) {
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
        auto p = PointAffine::load(points + gid * PointAffine::N_WORDS).to_projective();
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
        u32 *&d_points,
        u32 *&h_points_precompute,
        u32 &head, // d_points store head or tail
        cudaStream_t stream = 0
    ) {
        cudaError_t err;
        u64 part_len = div_ceil(len, Config::n_parts);
        head = 1;
        PROPAGATE_CUDA_ERROR(cudaHostAlloc(&h_points_precompute, sizeof(PointAffine) * len * Config::n_precompute, cudaHostAllocDefault));

        if constexpr (Config::n_precompute == 1) {
            PROPAGATE_CUDA_ERROR(cudaMallocAsync(&d_points, sizeof(PointAffine) * part_len, stream));
            PROPAGATE_CUDA_ERROR(cudaMemcpyAsync(d_points, h_points, sizeof(PointAffine) * part_len, cudaMemcpyHostToDevice, stream));
            memcpy(h_points_precompute, h_points, sizeof(PointAffine) * len);
            return cudaSuccess;
        }

        
        PROPAGATE_CUDA_ERROR(cudaMallocAsync(&d_points, sizeof(PointAffine) * part_len * Config::n_precompute, stream));

        for (int i = Config::n_parts - 1; i >= 0; i--) {
            u64 offset = i * part_len;
            u64 cur_len = std::min(part_len, len - offset);

            PROPAGATE_CUDA_ERROR(cudaMemcpyAsync(d_points, h_points + offset * PointAffine::N_WORDS, sizeof(PointAffine) * part_len, cudaMemcpyHostToDevice, stream));

            u32 grid = div_ceil(cur_len, 256);
            u32 block = 256;
            precompute_kernel<Config><<<grid, block, 0, stream>>>(d_points, cur_len);

            PROPAGATE_CUDA_ERROR(cudaMemcpyAsync(h_points_precompute + offset * Config::n_precompute * PointAffine::N_WORDS, d_points, sizeof(PointAffine) * part_len * Config::n_precompute, cudaMemcpyDeviceToHost, stream));
        }

        return cudaSuccess;
    }


    template <typename Config>
    __host__ cudaError_t run(
        const u32 *h_scalers,
        u32 *&d_points,
        u64 len,
        Point &h_result,
        const u32 *h_points_precompute,
        u32 &head,
        cudaStream_t stream = 0
    ) {
        cudaError_t err;

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

        cudaStream_t stream_copy; // stream for copy
        PROPAGATE_CUDA_ERROR(cudaStreamCreate(&stream_copy));

        cudaEvent_t scaler_copy_done, scaler_use_done, point_copy_done, point_use_done;
        PROPAGATE_CUDA_ERROR(cudaEventCreate(&scaler_copy_done));
        PROPAGATE_CUDA_ERROR(cudaEventCreate(&scaler_use_done));
        PROPAGATE_CUDA_ERROR(cudaEventCreate(&point_copy_done));
        PROPAGATE_CUDA_ERROR(cudaEventCreate(&point_use_done));

        PROPAGATE_CUDA_ERROR(cudaEventRecord(scaler_use_done, stream));
        PROPAGATE_CUDA_ERROR(cudaEventRecord(point_use_done, stream));


        Point *buckets_sum_buf;
        PROPAGATE_CUDA_ERROR(cudaMallocAsync(&buckets_sum_buf, sizeof(Point) * Config::n_windows * (Config::n_buckets - 1), stream));
        auto buckets_sum = Array2D<Point, Config::n_windows, Config::n_buckets-1>(buckets_sum_buf);
        
        unsigned short *mutex;
        PROPAGATE_CUDA_ERROR(cudaMallocAsync(&mutex, sizeof(unsigned short) * Config::n_buckets, stream));
        PROPAGATE_CUDA_ERROR(cudaMemsetAsync(mutex, 0, sizeof(unsigned short) * Config::n_buckets, stream));

        unsigned short *initialized_buf;
        PROPAGATE_CUDA_ERROR(cudaMallocAsync(&initialized_buf, sizeof(unsigned short) * (Config::n_buckets - 1) * Config::n_windows, stream));
        PROPAGATE_CUDA_ERROR(cudaMemsetAsync(initialized_buf, 0, sizeof(unsigned short) * (Config::n_buckets - 1) * Config::n_windows, stream));
        Array2D<unsigned short, Config::n_windows, Config::n_buckets - 1> initialized(initialized_buf);

        u32 *cnt_zero;
        PROPAGATE_CUDA_ERROR(cudaMallocAsync(&cnt_zero, sizeof(u32) * Config::n_windows, stream));         

        u32 h_points_per_window[Config::n_windows];
        u32 h_points_offset[Config::n_windows + 1];
        h_points_offset[0] = 0;
        #pragma unroll
        for (u32 i = 0; i < Config::n_windows; i++) {
            h_points_per_window[i] = div_ceil(Config::actual_windows - i, Config::n_windows);
            h_points_offset[i + 1] = h_points_offset[i] + h_points_per_window[i];
        }
        
        // points_offset[Config::n_windows] should be the total number of points
        assert(h_points_offset[Config::n_windows] == Config::actual_windows);

        u32 *d_points_offset;
        PROPAGATE_CUDA_ERROR(cudaMallocAsync(&d_points_offset, sizeof(u32) * (Config::n_windows + 1), stream));
        PROPAGATE_CUDA_ERROR(cudaMemcpyAsync(d_points_offset, h_points_offset, sizeof(u32) * (Config::n_windows + 1), cudaMemcpyHostToDevice, stream));

        u64 part_len = div_ceil(len, Config::n_parts);

        // for sorting, after sort, points with same bucket id are gathered, gives pointer to original index
        u64 *indexs;
        // window0 is the largest, so we use it size as buffer for sorting
        // possible overflow in bucket_sum does not matter for after sorting there are window0 points empty buffer in tail
        PROPAGATE_CUDA_ERROR(cudaMallocAsync(&indexs, sizeof(u64) * (part_len * (Config::actual_windows + h_points_per_window[0])), stream));

        u32 *scalers;
        PROPAGATE_CUDA_ERROR(cudaMallocAsync(&scalers, sizeof(u32) * Number::LIMBS * part_len, stream));

        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;

        cub::DeviceRadixSort::SortKeys(
            d_temp_storage, temp_storage_bytes,
            indexs + h_points_per_window[0] * part_len, indexs, h_points_per_window[0] * part_len, 0, Config::s, stream
        );

        PROPAGATE_CUDA_ERROR(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));

        int begin, end, stride;
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

        // bit 0 - Config::s for bucket id, the rest for point index
        // max log(2^30(max points) * precompute) + Config::s bits are needed
        static_assert(log2_ceil(Config::n_precompute) + 30 + Config::s <= 64, "Index too large");
        
        for (int p = begin; p != end; p += stride) {
            u64 offset = p * part_len;
            u64 cur_len = std::min(part_len, len - offset);

            PROPAGATE_CUDA_ERROR(cudaStreamWaitEvent(stream_copy, scaler_use_done, cudaEventWaitDefault));
            PROPAGATE_CUDA_ERROR(cudaMemcpyAsync(scalers, h_scalers + offset * Number::LIMBS, sizeof(u32) * Number::LIMBS * cur_len, cudaMemcpyHostToDevice, stream_copy));
            PROPAGATE_CUDA_ERROR(cudaEventRecord(scaler_copy_done, stream_copy));
            
            PROPAGATE_CUDA_ERROR(cudaMemsetAsync(cnt_zero, 0, sizeof(u32) * Config::n_windows, stream));

            PROPAGATE_CUDA_ERROR(cudaStreamWaitEvent(stream, scaler_copy_done, cudaEventWaitDefault));

            cudaEvent_t start, stop;
            float elapsedTime = 0.0;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            if constexpr (Config::debug) {
                cudaEventRecord(start, stream);
            }
            
            u32 block_size = 512;
            u32 grid_size = deviceProp.multiProcessorCount;
            distribute_windows<Config><<<grid_size, block_size, 0, stream>>>(scalers, cur_len, cnt_zero, indexs + h_points_per_window[0] * cur_len, d_points_offset);
            PROPAGATE_CUDA_ERROR(cudaDeviceSynchronize());

            PROPAGATE_CUDA_ERROR(cudaGetLastError());
            PROPAGATE_CUDA_ERROR(cudaEventRecord(scaler_use_done, stream));

            if constexpr (Config::debug) {
                cudaEventRecord(stop, stream);
                PROPAGATE_CUDA_ERROR(cudaEventSynchronize(stop));
                cudaEventElapsedTime(&elapsedTime, start, stop);
                std::cout << "MSM distribute_windows time:" << elapsedTime << std::endl;
            }
            // // print indexs and cnt_zero
            // u64* h_indexs = (u64*)malloc(sizeof(u64) * (Config::actual_windows + h_points_per_window[0]) * cur_len);
            // u32* h_cnt_zero = (u32*)malloc(sizeof(u32) * Config::n_windows);
            // PROPAGATE_CUDA_ERROR(cudaMemcpy(h_indexs, indexs, sizeof(u64) * (Config::actual_windows + h_points_per_window[0]) * cur_len, cudaMemcpyDeviceToHost));
            // PROPAGATE_CUDA_ERROR(cudaMemcpy(h_cnt_zero, cnt_zero, sizeof(u32) * Config::n_windows, cudaMemcpyDeviceToHost));
            // for (int i = 0; i < h_points_per_window[0] * cur_len; i++) {
            //     printf("indexs[%d]: key: %d\n", i, h_indexs[i + h_points_per_window[0] * cur_len] & ((1 << Config::s) - 1));
            // }
            // for (int i = 0; i < Config::n_windows; i++) {
            //     printf("cnt_zero[%d]: %d\n", i, h_cnt_zero[i]);
            // }
            if constexpr (Config::debug) {
                cudaEventRecord(start, stream);
            }

            for (int i = 0; i < Config::n_windows; i++) {
                cub::DeviceRadixSort::SortKeys(
                    d_temp_storage, temp_storage_bytes,
                    indexs + (h_points_per_window[0] + h_points_offset[i]) * cur_len,
                    indexs + h_points_offset[i] * cur_len, h_points_per_window[i] * cur_len, 0, Config::s, stream
                );
            }

            // PROPAGATE_CUDA_ERROR(cudaMemcpy(h_indexs, indexs, sizeof(u64) * (Config::actual_windows + h_points_per_window[0]) * cur_len, cudaMemcpyDeviceToHost));
            // PROPAGATE_CUDA_ERROR(cudaMemcpy(h_cnt_zero, cnt_zero, sizeof(u32) * Config::n_windows, cudaMemcpyDeviceToHost));
            // for (int i = 0; i < h_points_per_window[0] * cur_len; i++) {
            //     printf("indexs[%d]: key: %d\n", i, h_indexs[i] & ((1 << Config::s) - 1));
            // }
            // for (int i = 0; i < Config::n_windows; i++) {
            //     printf("cnt_zero[%d]: %d\n", i, h_cnt_zero[i]);
            // }

            PROPAGATE_CUDA_ERROR(cudaGetLastError());
            if constexpr (Config::debug) {
                cudaEventRecord(stop, stream);
                PROPAGATE_CUDA_ERROR(cudaEventSynchronize(stop));
                cudaEventElapsedTime(&elapsedTime, start, stop);
                std::cout << "MSM sort time:" << elapsedTime << std::endl;
            }

            PROPAGATE_CUDA_ERROR(cudaStreamWaitEvent(stream_copy, point_use_done, cudaEventWaitDefault));
            if (p != begin) {
                PROPAGATE_CUDA_ERROR(cudaMemcpyAsync(d_points, h_points_precompute + offset * Config::n_precompute * PointAffine::N_WORDS, sizeof(PointAffine) * cur_len * Config::n_precompute, cudaMemcpyHostToDevice, stream_copy));
            }
            PROPAGATE_CUDA_ERROR(cudaEventRecord(point_copy_done, stream_copy));


            PROPAGATE_CUDA_ERROR(cudaStreamWaitEvent(stream, point_copy_done, cudaEventWaitDefault));

            if constexpr (Config::debug) {
                cudaEventRecord(start, stream);
            }

            // Do bucket sum
            block_size = 256;
            grid_size = deviceProp.multiProcessorCount;

            for (u32 i = 0; i < Config::n_windows; i++) {
                bucket_sum<Config, 8><<<grid_size, block_size, 0, stream>>>(
                    i,
                    h_points_per_window[i] * cur_len,
                    indexs + h_points_offset[i] * cur_len,
                    d_points,
                    cnt_zero,
                    mutex,
                    initialized,
                    buckets_sum
                );
            }
            PROPAGATE_CUDA_ERROR(cudaGetLastError());
            PROPAGATE_CUDA_ERROR(cudaEventRecord(point_use_done, stream));

            if constexpr (Config::debug) {
                cudaEventRecord(stop, stream);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&elapsedTime, start, stop);
                std::cout << "MSM bucket sum time:" << elapsedTime << std::endl;
            }

        }

        PROPAGATE_CUDA_ERROR(cudaFreeAsync(mutex, stream));
        PROPAGATE_CUDA_ERROR(cudaFreeAsync(scalers, stream));
        PROPAGATE_CUDA_ERROR(cudaFreeAsync(d_points_offset, stream));
        PROPAGATE_CUDA_ERROR(cudaFreeAsync(d_temp_storage, stream));
        PROPAGATE_CUDA_ERROR(cudaFreeAsync(indexs, stream));
        PROPAGATE_CUDA_ERROR(cudaFreeAsync(cnt_zero, stream));

        cudaEvent_t start, stop;
        float ms;

        if constexpr (Config::debug) {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
        }

        u32 grid = div_ceil(deviceProp.multiProcessorCount, Config::n_windows) * Config::n_windows; 
        Point *reduce_buffer;
        PROPAGATE_CUDA_ERROR(cudaMallocAsync(&reduce_buffer, sizeof(Point) * Config::n_windows * grid, stream));

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

        // ruduce all
        void *d_temp_storage_reduce = nullptr;
        mont::usize temp_storage_bytes_reduce = 0;

        Point *reduced;
        PROPAGATE_CUDA_ERROR(cudaMallocAsync(&reduced, sizeof(Point), stream));

        auto add_op = [] __device__ __host__(const Point &a, const Point &b) { return a + b; };

        PROPAGATE_CUDA_ERROR(
            cub::DeviceReduce::Reduce(
                d_temp_storage_reduce, 
                temp_storage_bytes_reduce, 
                reduce_buffer, 
                reduced, 
                Config::n_windows * grid, 
                add_op,
                Point::identity(),
                stream
            )
        );

        PROPAGATE_CUDA_ERROR(cudaMallocAsync(&d_temp_storage_reduce, temp_storage_bytes_reduce, stream));

        if constexpr (Config::debug) {
            cudaEventRecord(start, stream);
        }

        PROPAGATE_CUDA_ERROR(
            cub::DeviceReduce::Reduce(
                d_temp_storage_reduce, 
                temp_storage_bytes_reduce, 
                reduce_buffer, 
                reduced, 
                Config::n_windows * grid,
                add_op,
                Point::identity(),
                stream
            )
        );

        if constexpr (Config::debug) {
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            std::cout << "MSM final reduce time:" << ms << std::endl;
        }

        PROPAGATE_CUDA_ERROR(cudaFreeAsync(d_temp_storage_reduce, stream));
        PROPAGATE_CUDA_ERROR(cudaFreeAsync(reduce_buffer, stream));

        PROPAGATE_CUDA_ERROR(cudaMemcpyAsync(&h_result, reduced, sizeof(Point), cudaMemcpyDeviceToHost, stream));
        PROPAGATE_CUDA_ERROR(cudaFreeAsync(reduced, stream));

        return cudaSuccess;
    }

  }
