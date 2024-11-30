#include "../../mont/src/bn254_scalar.cuh"
#include "bn254.cuh"

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

namespace msm
{
  using bn254::Point;
  using bn254::PointAffine;
  using bn254_scalar::Element;
  using bn254_scalar::Number;
  using mont::u32;
  using mont::u64;

  const u32 THREADS_PER_WARP = 32;

  constexpr u32 pow2(u32 n)
  {
    return n == 0 ? 1 : 2 * pow2(n - 1);
  }

  constexpr __forceinline__ __host__ __device__ u32 div_ceil(u32 a, u32 b)
  {
    return (a + b - 1) / b;
  }

  constexpr int log2_floor(int n)
  {
    return (n == 1) ? 0 : 1 + log2_floor(n / 2);
  }

  constexpr int log2_ceil(int n)
  {
    // Check if n is a power of 2
    if ((n & (n - 1)) == 0)
    {
      return log2_floor(n);
    }
    else
    {
      return 1 + log2_floor(n);
    }
  }

  template <typename T, u32 D1, u32 D2>
  struct Array2D
  {
    T *buf;

    Array2D(T *buf) : buf(buf) {}
    __host__ __device__ __forceinline__ T &get(u32 i, u32 j)
    {
      return buf[i * D2 + j];
    }
    __host__ __device__ __forceinline__ const T &get_const(u32 i, u32 j) const
    {
      return buf[i * D2 + j];
    }
    __host__ __device__ __forceinline__ T *addr(u32 i, u32 j)
    {
      return buf + i * D2 + j;
    }
  };

  struct MsmConfig
  {
    static constexpr u32 lambda = 256;
    static constexpr u32 s = 16;
    static constexpr u32 n_windows = div_ceil(lambda, s);
    static constexpr u32 n_buckets = pow2(s);
    static constexpr u32 n_window_id_bits = log2_ceil(n_windows);

    static constexpr u32 scatter_batch_size = 32;

    static constexpr bool debug = false;
  };

  template <typename Config>
  __global__ void count_buckets(
      const u32 *scalers,
      const u32 len,
      u32* cnt_zero,
      u32* keys,
      u32* values)
  {
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    u32 stride = gridDim.x * blockDim.x;
    
    // Count into block-wide counter
    for (u32 i = tid; i < len; i += stride) {
      u32 bucket[Config::n_windows];
      auto scaler = Number::load(scalers + i * Number::LIMBS);
      scaler.bit_slice<Config::n_windows, Config::s>(bucket);

      #pragma unroll
      for (u32 window_id = 0; window_id < Config::n_windows; window_id++) {
        auto bucket_id = bucket[window_id];
        if (bucket_id == 0) atomicAdd(cnt_zero + window_id, 1);
        keys[window_id * len + i] = bucket_id;
        values[window_id * len + i] = i;
      }
    }
  }
    template <typename Config, typename Point>
    __global__ void initalize_sum(Array2D<Point, Config::n_windows, Config::n_buckets - 1> sum) {
        u32 bucket_id = blockIdx.x;
        u32 window_id = threadIdx.x;

        if (bucket_id >= Config::n_buckets || window_id >= Config::n_windows)
            return;

        sum.get(window_id, bucket_id) = Point::identity();
    }

  template <typename Config, u32 WarpPerBlock>
  __global__ void bucket_sum(
      const u32 window_id,
      const u32 len,
      const u32 *key_for_pointer,
      const u32 *pointer_to_points,
      const u32 *points,
      const u32 *cnt_zero,
      u32 *mutex,
      Array2D<Point, Config::n_windows, Config::n_buckets - 1> sum)
  {

    const u32 gtid = threadIdx.x + blockIdx.x * blockDim.x;
    const u32 threads = blockDim.x * gridDim.x;
    const u32 zero_offset = cnt_zero[window_id];
    const u32 work_len = div_ceil(len - zero_offset, threads);
    const u32 start_id = work_len * gtid;
    const u32 end_id = min(start_id + work_len, len - zero_offset);
    if (start_id >= end_id) return;
    key_for_pointer += zero_offset;
    pointer_to_points += zero_offset;

    auto acc = Point::identity();
    u32 last_key = key_for_pointer[start_id];

    for (u32 i = start_id; i < end_id; i++) {
      u32 cur_key = key_for_pointer[i];
      if (cur_key != last_key) {
        auto mutex_ptr = mutex + last_key;
        u32 time = 1;
        while (atomicCAS(mutex_ptr, (unsigned short int)0, (unsigned short int)1) != 0) {
            __nanosleep(time);
            if (time < 16) time *= 2;
        }
        __threadfence();
        sum.get(window_id, last_key - 1) = sum.get(window_id, last_key - 1) + acc;
        __threadfence();
        atomicCAS(mutex_ptr, (unsigned short int)1, (unsigned short int)0);
        acc = Point::identity();
      }
      last_key = cur_key;
      auto pointer = pointer_to_points[i];
      auto p = PointAffine::load(points + pointer * PointAffine::N_WORDS);
      acc = acc + p;
    }
    auto mutex_ptr = mutex + last_key;
    u32 time = 1;
    while (atomicCAS(mutex_ptr, (unsigned short int)0, (unsigned short int)1) != 0) {
        __nanosleep(time);
        if (time < 16) time *= 2;
    }
    __threadfence();
    sum.get(window_id, last_key - 1) = sum.get(window_id, last_key - 1) + acc;
    __threadfence();
    atomicCAS(mutex_ptr, (unsigned short int)1, (unsigned short int)0);
  }

    template<typename Config, u32 blockdim = 256>
    __launch_bounds__(256,1)
    __global__ void reduceBuckets(Array2D<Point, Config::n_windows, Config::n_buckets - 1> buckets_sum, Point *reduceMemory) {

        assert(gridDim.x % Config::n_windows == 0);

        using BlockReduce = cub::BlockReduce<Point, blockdim>;
        __shared__ typename BlockReduce::TempStorage temp_storage;

        const u32 total_threads_per_window = gridDim.x / Config::n_windows * blockDim.x;
        u32 window_id = blockIdx.x / (gridDim.x / Config::n_windows);

        u32 wtid = (blockIdx.x % (gridDim.x / Config::n_windows)) * blockDim.x + threadIdx.x;
        
        const u32 buckets_per_thread = div_ceil(Config::n_buckets - 1, total_threads_per_window);

        Point sum, sum_of_sums;

        sum = Point::identity();
        sum_of_sums = Point::identity();

        for(u32 i=buckets_per_thread; i > 0; i--) {
            u32 loadIndex = wtid * buckets_per_thread + i;
            if(loadIndex < Config::n_buckets) {
                sum = sum + buckets_sum.get(window_id, loadIndex - 1);
            }
            sum_of_sums = sum_of_sums + sum;
        }

        u32 scale = wtid * buckets_per_thread;

        sum = sum.multiple(scale);

        sum_of_sums = sum_of_sums + sum;

        auto aggregate = BlockReduce(temp_storage).Reduce(sum_of_sums, [](const Point &a, const Point &b){ return a + b; });

        if (threadIdx.x == 0) {
            for (u32 i = 0; i < window_id * Config::s; i++) {
                aggregate = aggregate.self_add();
            }
            reduceMemory[blockIdx.x] = aggregate;
        }
    }


  template <typename Config>
  __host__ cudaError_t run(
      const u32 *h_scalers,
      const u32 *h_points,
      u32 len,
      Point &h_result)
  {
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
    
    u32 *keys, *values;

    PROPAGATE_CUDA_ERROR(cudaMalloc(&keys, sizeof(u32) * len * Config::n_windows));
    PROPAGATE_CUDA_ERROR(cudaMalloc(&values, sizeof(u32) * len * Config::n_windows));

    PROPAGATE_CUDA_ERROR(cudaMemcpy(scalers, h_scalers, sizeof(u32) * Number::LIMBS * len, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    u32 block_size = 512;
    u32 grid_size = deviceProp.multiProcessorCount * 4;
    count_buckets<Config><<<grid_size, block_size>>>(scalers, len, cnt_zero, keys, values);

    PROPAGATE_CUDA_ERROR(cudaGetLastError());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "MSM count time:" << elapsedTime << std::endl;

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    

    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        keys, keys, values, values, len
    );

    PROPAGATE_CUDA_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    elapsedTime = 0.0;
    cudaEventRecord(start);

    for (int i = 0; i < Config::n_windows; i++) {
        cub::DeviceRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            keys + i * len, keys + i * len, values + i * len, values + i * len, len
        );
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "MSM sort time:" << elapsedTime << std::endl;

    PROPAGATE_CUDA_ERROR(cudaFree(d_temp_storage));
    PROPAGATE_CUDA_ERROR(cudaFree(scalers));
    
    d_temp_storage = nullptr;
    temp_storage_bytes = 0;
    
    // Prepare for bucket sum
    Point *buckets_sum_buf;
    PROPAGATE_CUDA_ERROR(cudaMalloc(&buckets_sum_buf, sizeof(Point) * Config::n_windows * (Config::n_buckets - 1)));
    auto buckets_sum = Array2D<Point, Config::n_windows, Config::n_buckets-1>(buckets_sum_buf);

    // TODO: Cut into chunks to support len at 2^30
    u32 *points;
    PROPAGATE_CUDA_ERROR(cudaMalloc(&points, sizeof(u32) * PointAffine::N_WORDS * len));
    PROPAGATE_CUDA_ERROR(cudaMemcpy(points, h_points, sizeof(u32) * PointAffine::N_WORDS * len, cudaMemcpyHostToDevice));   

    u32 *mutex;
    PROPAGATE_CUDA_ERROR(cudaMalloc(&mutex, sizeof(u32) * Config::n_buckets));
    PROPAGATE_CUDA_ERROR(cudaMemset(mutex, 0, sizeof(u32) * Config::n_buckets));
    initalize_sum<Config, Point><<<Config::n_buckets - 1, Config::n_windows>>>(buckets_sum);

    // Do bucket sum
    cudaEventRecord(start);
    block_size = 256;
    grid_size = deviceProp.multiProcessorCount;

    for (u32 i = 0; i < Config::n_windows; i++) {
      bucket_sum<Config, 8><<<grid_size, block_size>>>(
        i,
        len,
        keys + i * len,
        values + i * len,
        points,
        cnt_zero,
        mutex,
        buckets_sum
      );
    }
    
    PROPAGATE_CUDA_ERROR(cudaGetLastError());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "MSM bucket sum time:" << elapsedTime << std::endl;

    PROPAGATE_CUDA_ERROR(cudaGetLastError());
    PROPAGATE_CUDA_ERROR(cudaFree(points));
    PROPAGATE_CUDA_ERROR(cudaFree(mutex));
    PROPAGATE_CUDA_ERROR(cudaFree(values));
    PROPAGATE_CUDA_ERROR(cudaFree(keys));
    PROPAGATE_CUDA_ERROR(cudaFree(cnt_zero));

    float ms;
    cudaEvent_t end;
    cudaEventCreate(&end);
    u32 grid = deviceProp.multiProcessorCount;

    Point *reduce_buffer;

        PROPAGATE_CUDA_ERROR(cudaMalloc(&reduce_buffer, sizeof(Point) * Config::n_windows * grid));


        // start reduce

        PROPAGATE_CUDA_ERROR(cudaEventRecord(start));
        
        reduceBuckets<Config, 256> <<< grid * Config::n_windows, 256 >>> (buckets_sum, reduce_buffer);

        PROPAGATE_CUDA_ERROR(cudaEventRecord(end));
        PROPAGATE_CUDA_ERROR(cudaEventSynchronize(end));
        PROPAGATE_CUDA_ERROR(cudaEventElapsedTime(&ms, start, end));
        printf("bucket reduce: %f ms\n", ms);

        PROPAGATE_CUDA_ERROR(cudaFree(buckets_sum_buf));

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
