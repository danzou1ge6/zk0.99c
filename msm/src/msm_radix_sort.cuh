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

        static constexpr bool debug = false;
    };

    // divide scalers into windows
    // count number of zeros in each window
    template <typename Config>
    __global__ void distribute_windows(
        const u32 *scalers,
        const u64 len,
        u32* cnt_zero,
        u64* indexs,
        u64* points_offset
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
                if (bucket_id == 0) atomicAdd(cnt_zero + physical_window_id, 1);
                indexs[points_offset[physical_window_id] + len * point_group + i] = bucket_id | ((point_group * len + i) << Config::s);
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

//      printf("pointer: %d\n", pointer);
//      printf("key: %d\n", key);
//      Point tmp = Point::identity();
//      tmp = tmp + p;
//      for (u32 j = 0; j < window_id * Config::s; j++) {
//         tmp = tmp.self_add();
//      }
//      auto pp = tmp.to_affine();
//      for (u32 j = 0; j < 8; j++) {
//          printf("p[%d]: %d ", j, pp.x.n.limbs[j]);
//      }
//      printf("\n");

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
        u32 *&d_points
    ) {
        cudaError_t err;
        PROPAGATE_CUDA_ERROR(cudaMalloc(&d_points, sizeof(PointAffine) * len * Config::n_precompute));
        PROPAGATE_CUDA_ERROR(cudaMemcpy(d_points, h_points, sizeof(PointAffine) * len, cudaMemcpyHostToDevice));
        if (Config::n_precompute > 1) {
            u32 grid = div_ceil(len, 256);
            u32 block = 256;
            precompute_kernel<Config><<<grid, block>>>(d_points, len);
            PROPAGATE_CUDA_ERROR(cudaDeviceSynchronize());
        }
        return cudaSuccess;
    }


    template <typename Config>
    __host__ cudaError_t run(
        const u32 *h_scalers,
        const u32 *d_points,
        u64 len,
        Point &h_result
    ) {
        cudaError_t err;

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

        // Count items in buckets
        u32 *scalers;
        PROPAGATE_CUDA_ERROR(cudaMalloc(&scalers, sizeof(u32) * Number::LIMBS * len));
        
        u32 *cnt_zero;
        PROPAGATE_CUDA_ERROR(cudaMalloc(&cnt_zero, sizeof(u32) * Config::n_windows));
        PROPAGATE_CUDA_ERROR(cudaMemset(cnt_zero, 0, sizeof(u32) * Config::n_windows));

        // for sorting, after sort, points with same bucket id are gathered, gives pointer to original index
        
        // bit 0 - Config::s for bucket id, the rest for point index
        // max log(2^30(max points) * precompute) + Config::s bits are needed
        static_assert(log2_ceil(Config::n_precompute) + 30 + Config::s <= 64, "Index too large");
        u64 *indexs;

        u64 h_points_per_window[Config::n_windows];
        u64 h_points_offset[Config::n_windows + 1];
        h_points_offset[0] = 0;
        #pragma unroll
        for (u32 i = 0; i < Config::n_windows; i++) {
            h_points_per_window[i] = div_ceil(Config::actual_windows - i, Config::n_windows) * len;
            h_points_offset[i + 1] = h_points_offset[i] + h_points_per_window[i];
        }
        
        // points_offset[Config::n_windows] should be the total number of points
        assert(h_points_offset[Config::n_windows] == Config::actual_windows * len);

        u64 *d_points_offset;
        PROPAGATE_CUDA_ERROR(cudaMalloc(&d_points_offset, sizeof(u64) * (Config::n_windows + 1)));
        PROPAGATE_CUDA_ERROR(cudaMemcpy(d_points_offset, h_points_offset, sizeof(u64) * (Config::n_windows + 1), cudaMemcpyHostToDevice));

        // window0 is the largest, so we use it size as buffer for sorting
        // possible overflow in bucket_sum does not matter for after sorting there are window0 points empty buffer in tail
        PROPAGATE_CUDA_ERROR(cudaMalloc(&indexs, sizeof(u64) * (len * (Config::actual_windows) + h_points_per_window[0])));

        PROPAGATE_CUDA_ERROR(cudaMemcpy(scalers, h_scalers, sizeof(u32) * Number::LIMBS * len, cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        float elapsedTime = 0.0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        u32 block_size = 512;
        u32 grid_size = deviceProp.multiProcessorCount * 4;
        distribute_windows<Config><<<grid_size, block_size>>>(scalers, len, cnt_zero, indexs + h_points_per_window[0], d_points_offset);

        PROPAGATE_CUDA_ERROR(cudaGetLastError());
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        std::cout << "MSM distribute_windows time:" << elapsedTime << std::endl;
        
        PROPAGATE_CUDA_ERROR(cudaFree(scalers));
        PROPAGATE_CUDA_ERROR(cudaFree(d_points_offset));

        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        

        cub::DeviceRadixSort::SortKeys(
            d_temp_storage, temp_storage_bytes,
            indexs + h_points_per_window[0], indexs, h_points_per_window[0], 0, Config::s
        );

        PROPAGATE_CUDA_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        
        elapsedTime = 0.0;
        cudaEventRecord(start);

        for (int i = 0; i < Config::n_windows; i++) {
            cub::DeviceRadixSort::SortKeys(
                d_temp_storage, temp_storage_bytes,
                indexs + (h_points_per_window[0] + h_points_offset[i]), indexs + h_points_offset[i], h_points_per_window[i], 0, Config::s
            );
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        std::cout << "MSM sort time:" << elapsedTime << std::endl;

        PROPAGATE_CUDA_ERROR(cudaFree(d_temp_storage));
        
        d_temp_storage = nullptr;
        temp_storage_bytes = 0;
        
        // Prepare for bucket sum
        Point *buckets_sum_buf;
        PROPAGATE_CUDA_ERROR(cudaMalloc(&buckets_sum_buf, sizeof(Point) * Config::n_windows * (Config::n_buckets - 1)));
        auto buckets_sum = Array2D<Point, Config::n_windows, Config::n_buckets-1>(buckets_sum_buf);

        unsigned short *mutex;
        PROPAGATE_CUDA_ERROR(cudaMalloc(&mutex, sizeof(unsigned short) * Config::n_buckets));
        PROPAGATE_CUDA_ERROR(cudaMemset(mutex, 0, sizeof(unsigned short) * Config::n_buckets));

        unsigned short *initialized_buf;
        PROPAGATE_CUDA_ERROR(cudaMalloc(&initialized_buf, sizeof(unsigned short) * (Config::n_buckets - 1) * Config::n_windows));
        PROPAGATE_CUDA_ERROR(cudaMemset(initialized_buf, 0, sizeof(unsigned short) * (Config::n_buckets - 1) * Config::n_windows));
        Array2D<unsigned short, Config::n_windows, Config::n_buckets - 1> initialized(initialized_buf);

        // Do bucket sum
        cudaEventRecord(start);
        block_size = 256;
        grid_size = deviceProp.multiProcessorCount;

        for (u32 i = 0; i < Config::n_windows; i++) {
            bucket_sum<Config, 8><<<grid_size, block_size>>>(
                i,
                h_points_per_window[i],
                indexs + h_points_offset[i],
                d_points,
                cnt_zero,
                mutex,
                initialized,
                buckets_sum
            );
        }
        
        PROPAGATE_CUDA_ERROR(cudaGetLastError());
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        std::cout << "MSM bucket sum time:" << elapsedTime << std::endl;

        PROPAGATE_CUDA_ERROR(cudaGetLastError());
        PROPAGATE_CUDA_ERROR(cudaFree(mutex));
        PROPAGATE_CUDA_ERROR(cudaFree(indexs));
        PROPAGATE_CUDA_ERROR(cudaFree(cnt_zero));

        float ms;
        cudaEvent_t end;
        cudaEventCreate(&end);
        u32 grid = div_ceil(deviceProp.multiProcessorCount, Config::n_windows) * Config::n_windows; 

        Point *reduce_buffer;

        PROPAGATE_CUDA_ERROR(cudaMalloc(&reduce_buffer, sizeof(Point) * Config::n_windows * grid));

        // start reduce

        PROPAGATE_CUDA_ERROR(cudaEventRecord(start));
        
        reduceBuckets<Config, 8> <<< grid, 256 >>> (buckets_sum, reduce_buffer, initialized);

        PROPAGATE_CUDA_ERROR(cudaEventRecord(end));
        PROPAGATE_CUDA_ERROR(cudaEventSynchronize(end));
        PROPAGATE_CUDA_ERROR(cudaEventElapsedTime(&ms, start, end));
        printf("bucket reduce: %f ms\n", ms);

        PROPAGATE_CUDA_ERROR(cudaFree(buckets_sum_buf));
        PROPAGATE_CUDA_ERROR(cudaFree(initialized_buf));

        // ruduce all
        void *d_temp_storage_reduce = nullptr;
        mont::usize temp_storage_bytes_reduce = 0;

        Point *reduced;
        PROPAGATE_CUDA_ERROR(cudaMalloc(&reduced, sizeof(Point)));

        auto add_op = [] __device__ __host__(const Point &a, const Point &b) { return a + b; };

        PROPAGATE_CUDA_ERROR(
            cub::DeviceReduce::Reduce(
                d_temp_storage_reduce, 
                temp_storage_bytes_reduce, 
                reduce_buffer, 
                reduced, 
                Config::n_windows * grid, 
                add_op,
                Point::identity()
            )
        );

        PROPAGATE_CUDA_ERROR(cudaMalloc(&d_temp_storage_reduce, temp_storage_bytes_reduce));

        PROPAGATE_CUDA_ERROR(cudaEventRecord(start));
        PROPAGATE_CUDA_ERROR(
            cub::DeviceReduce::Reduce(
                d_temp_storage_reduce, 
                temp_storage_bytes_reduce, 
                reduce_buffer, 
                reduced, 
                Config::n_windows * grid,
                add_op,
                Point::identity()
            )
        );
        PROPAGATE_CUDA_ERROR(cudaEventRecord(end));
        PROPAGATE_CUDA_ERROR(cudaEventSynchronize(end));
        PROPAGATE_CUDA_ERROR(cudaEventElapsedTime(&ms, start, end));
        printf("reduce: %f ms\n", ms);

        PROPAGATE_CUDA_ERROR(cudaFree(d_temp_storage_reduce));
        PROPAGATE_CUDA_ERROR(cudaFree(reduce_buffer));

        PROPAGATE_CUDA_ERROR(cudaMemcpy(&h_result, reduced, sizeof(Point), cudaMemcpyDeviceToHost));
        PROPAGATE_CUDA_ERROR(cudaFree(reduced));

        return cudaSuccess;
    }

  }
