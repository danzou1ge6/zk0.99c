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
      Array2D<u32, Config::n_windows, Config::n_buckets> counts)
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
        auto bucketid = bucket[window_id];
        if (bucketid != 0) {
          atomicAdd(counts.addr(window_id, bucketid), 1);
        }
      }
    }
  }

  template <typename Config>
  __global__ void scatter(
      const u32 *scalers,
      const u32 len,
      u32 *buckets_buffer,
      const Array2D<u32, Config::n_windows, Config::n_buckets> buckets_offset,
      Array2D<u32, Config::n_windows, Config::n_buckets> buckets_len)
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
        auto bucketid = bucket[window_id];
        if (bucketid != 0) {
          buckets_buffer[buckets_offset.get_const(window_id, bucketid) + atomicAdd(buckets_len.addr(window_id, bucketid), 1)] = i;
        }
      }
    }
  }

  template <typename Config, u32 WarpPerBlock>
  __global__ void bucket_sum(
      const u32 *buckets_buffer,
      const Array2D<u32, Config::n_windows, Config::n_buckets> buckets_offset,
      const Array2D<u32, Config::n_windows, Config::n_buckets> buckets_len,
      const u32 *points,
      Array2D<Point, Config::n_windows, Config::n_buckets - 1> sum)
  {

    u32 gtid = threadIdx.x + blockIdx.x * blockDim.x;

    u32 window_id = gtid / (Config::n_buckets - 1);
    u32 bucket_id = (gtid % (Config::n_buckets - 1)) + 1;

    if (bucket_id >= Config::n_buckets || window_id >= Config::n_windows)
      return;

    buckets_buffer += buckets_offset.get_const(window_id, bucket_id);

    u32 len = buckets_len.get_const(window_id, bucket_id);

    auto acc = Point::identity();

    for (u32 i = 0; i < len; i++) {
      auto p = PointAffine::load(points + buckets_buffer[i] * PointAffine::N_WORDS);
      acc = acc + p;
    }

    sum.get(window_id, bucket_id - 1) = acc;
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
    
    u32 *counts_buf;
    PROPAGATE_CUDA_ERROR(cudaMalloc(&counts_buf, sizeof(u32) * Config::n_windows * Config::n_buckets));
    PROPAGATE_CUDA_ERROR(cudaMemset(counts_buf, 0, sizeof(u32) * Config::n_windows * Config::n_buckets));
    PROPAGATE_CUDA_ERROR(cudaMemcpy(scalers, h_scalers, sizeof(u32) * Number::LIMBS * len, cudaMemcpyHostToDevice));
    auto counts = Array2D<u32, Config::n_windows, Config::n_buckets>(counts_buf);
    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    u32 block_size = 512;
    u32 grid_size = deviceProp.multiProcessorCount * 4;
    count_buckets<Config><<<grid_size, block_size>>>(scalers, len, counts); 
    PROPAGATE_CUDA_ERROR(cudaGetLastError());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "MSM count time:" << elapsedTime << std::endl;

    // Allocate space for buckets buffer
    // Space for PoindId's
    u32 *buckets_buffer;
    PROPAGATE_CUDA_ERROR(cudaMalloc(&buckets_buffer, sizeof(u32) * len * Config::n_windows));

    // Initialize bucket offsets
    u32 *buckets_offset_buf;
    PROPAGATE_CUDA_ERROR(cudaMalloc(&buckets_offset_buf, sizeof(u32) * Config::n_windows * Config::n_buckets));
    elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        counts_buf, buckets_offset_buf, Config::n_windows * Config::n_buckets);
    // Allocate temporary storage
    PROPAGATE_CUDA_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        counts_buf, buckets_offset_buf, Config::n_windows * Config::n_buckets);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "MSM exclusiveSum time:" << elapsedTime << std::endl;
    PROPAGATE_CUDA_ERROR(cudaFree(d_temp_storage));
    auto buckets_offset = Array2D<u32, Config::n_windows, Config::n_buckets>(buckets_offset_buf);
    

    elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // Scatter
    block_size = 512;
    grid_size = deviceProp.multiProcessorCount * 4;
    cudaMemset(counts_buf, 0, sizeof(u32) * Config::n_windows * Config::n_buckets);
    scatter<Config><<<grid_size, block_size>>>(
        scalers,
        len,
        buckets_buffer,
        buckets_offset,
        counts);
    PROPAGATE_CUDA_ERROR(cudaGetLastError());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "MSM scatter time:" << elapsedTime << std::endl;
    // space for counts is reused
    PROPAGATE_CUDA_ERROR(cudaFree(scalers));


    // Prepare for bucket sum
    Point *buckets_sum_buf;
    PROPAGATE_CUDA_ERROR(cudaMalloc(&buckets_sum_buf, sizeof(Point) * Config::n_windows * (Config::n_buckets - 1)));
    auto buckets_sum = Array2D<Point, Config::n_windows, Config::n_buckets-1>(buckets_sum_buf);

    // TODO: Cut into chunks to support len at 2^30
    u32 *points;
    PROPAGATE_CUDA_ERROR(cudaMalloc(&points, sizeof(u32) * PointAffine::N_WORDS * len));
    PROPAGATE_CUDA_ERROR(cudaMemcpy(points, h_points, sizeof(u32) * PointAffine::N_WORDS * len, cudaMemcpyHostToDevice));   

    // Do bucket sum
    cudaEventRecord(start);
    block_size = 256;
    grid_size = div_ceil((Config::n_buckets-1) * Config::n_windows, block_size);
    bucket_sum<Config, 8><<<grid_size, block_size>>>(
        buckets_buffer,
        buckets_offset,
        counts,
        points,
        buckets_sum
    );
    PROPAGATE_CUDA_ERROR(cudaGetLastError());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "MSM bucket sum time:" << elapsedTime << std::endl;

    PROPAGATE_CUDA_ERROR(cudaGetLastError());
    PROPAGATE_CUDA_ERROR(cudaFree(points));
    PROPAGATE_CUDA_ERROR(cudaFree(counts_buf));
    PROPAGATE_CUDA_ERROR(cudaFree(buckets_buffer));
    PROPAGATE_CUDA_ERROR(cudaFree(buckets_offset_buf));


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
