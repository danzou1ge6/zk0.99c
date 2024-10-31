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

  constexpr u32 div_ceil(u32 a, u32 b)
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
    static constexpr u32 s = 8;
    static constexpr u32 n_windows = div_ceil(lambda, s);
    static constexpr u32 n_buckets = pow2(s);
    static constexpr u32 n_window_id_bits = log2_ceil(n_windows);

    static constexpr u32 scatter_batch_size = 32;

    static constexpr bool debug = false;
  };

  // initialize the array 'counts'
  template <typename Config>
  __global__ void initialize_counts(Array2D<u32, Config::n_windows, Config::n_buckets> counts)
  {
    if (threadIdx.x < Config::n_windows)
    {
      u32 window_id = threadIdx.x;
      for (u32 i = 0; i < Config::n_buckets; i++)
      {
        counts.get(window_id, i) = 0;
      }
    }

    __syncthreads();

    // for(u32 i=0; i < Config::n_windows; ++i)
    // {
    //   for (u32 j = 0; j < Config::n_buckets; j++)
    //   {
    //     if(counts.get(i, j) != 0)
    //     {
    //       printf("Assertion failed! i: %d, j: %d, value: %d\n",
    //             i, j, counts.get(i, j));
    //       assert(false);
    //     }
    //   }
    // }
  }

  // with n_windows = 32, launched 64 threads
  //
  //   scaler 0 = digit 0 | d1 | ... | d31
  //               ^thread 0 ^t1        ^t31
  //   scaler 1 = digit 0 | d1 | ... | d31
  //               ^t32     ^t33        ^t63
  //   scaler 2 = digit 0 | d1 | ... | d31
  //               ^t0      ^t1         ^t31
  //   ...
  //
  // blockDim.x better be multiple of n_windows, and total number threads must be mutiple of n_windows
  template <typename Config>
  __global__ void count_buckets(
      const u32 *scalers,
      const u32 len,
      Array2D<u32, Config::n_windows, Config::n_buckets> counts)
  {
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    u32 window_id = tid % Config::n_windows;
    u32 i0_scaler = tid / Config::n_windows;

    u32 n_threads = gridDim.x * blockDim.x;
    u32 i_stride = n_threads / Config::n_windows;

    __shared__ u32 shm_counts[Config::n_windows][Config::n_buckets];

    // Initialize counters
    if (threadIdx.x < Config::n_windows)
    {
      u32 window_id = threadIdx.x;
      for (u32 i = 0; i < Config::n_buckets; i++)
      {
        shm_counts[window_id][i] = 0;

        // if (blockIdx.x == 0)
        //   counts.get(window_id, i) = 0;
      }
    }

    __syncthreads();

    // for(u32 i=0; i < Config::n_windows; ++i)
    // {
    //   for (u32 j = 0; j < Config::n_buckets; j++)
    //   {
    //     if(shm_counts[i][j] != 0)
    //     {
    //       printf("Assertion failed! i: %d, j: %d, value: %d\n",
    //             i, j, shm_counts[i][j]);
    //       assert(false);
    //     }
    //   }
    // }

    // Count into block-wide counter
    for (u32 i = i0_scaler; i < len; i += i_stride)
    {
      auto scaler = Number::load(scalers + i * Number::LIMBS);
      auto scaler_window = scaler.bit_slice(window_id * Config::s, Config::s);
      if (scaler_window != 0)
        atomicAdd(&shm_counts[window_id][scaler_window], 1);
    }

    __syncthreads();

    // Merge block-wide counter into global counter
    // Opt. Opportunity: Using cub::DeviceReudce instead
    if (threadIdx.x < Config::n_windows)
    {
      u32 window_id = threadIdx.x;
      for (u32 i = 0; i < Config::n_buckets; i++)
      {
        atomicAdd(&counts.get(window_id, i), shm_counts[window_id][i]);
      }
    }
  }

  struct PointId
  {
    u32 x;

    __device__ __forceinline__ PointId(u32 window_id, u32 scaler_id)
    {
      x = scaler_id;
    }

    __device__ __forceinline__ u32 scaler_id() const
    {
      return x;
    }
  };

  // initialize the array 'buckets_len'
  template <typename Config>
  __global__ void initialize_buckets_len(Array2D<u32, Config::n_windows, Config::n_buckets> buckets_len)
  {
    if (threadIdx.x < Config::n_windows)
    {
      for (u32 i = 0; i < Config::n_buckets; i++)
        buckets_len.get(threadIdx.x, i) = 0;
    }
  }

  // with n_windows = 32, scatter_batch_size = 2, launched 64 threads
  //
  //   scaler 0 = digit 0 | d1 | ... | d31
  //               ^thread 0 ^t1        ^t31
  //   scaler 1 = digit 0 | d1 | ... | d31
  //               ^t0       ^t1        ^t31
  //   scaler 2 = digit 0 | d1 | ... | d31
  //               ^t32      ^t33       ^t63
  //   scaler 3 = digit 0 | d1 | ... | d31
  //               ^t32      ^t33       ^t63
  //   ...
  //
  // blockDim.x better be multiple of n_windows, and total number threads must be mutiple of n_windows
  template <typename Config>
  __global__ void scatter(
      const u32 *scalers,
      const u32 len,
      PointId *buckets_buffer,
      const Array2D<u32, Config::n_windows, Config::n_buckets> buckets_offset,
      Array2D<u32, Config::n_windows, Config::n_buckets> buckets_len)
  {
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    u32 window_id = tid % Config::n_windows;
    u32 i0_scaler = tid / Config::n_windows * Config::scatter_batch_size;

    u32 n_threads = gridDim.x * blockDim.x;
    u32 i_stride = n_threads / Config::n_windows * Config::scatter_batch_size;

    // if (tid == 0)
    //   printf("config info: %d %d %d %d\n", Config::n_windows, Config::n_buckets, Config::scatter_batch_size, i_stride);

    // Opt. Opportunity: First scatter to shared memory, then scatter to global, so as to reduce
    // global atomic operations.
    for (u32 i = i0_scaler; i < len; i += i_stride)
    {
      for (u32 j = i; j < i + Config::scatter_batch_size && j < len; j++)
      {
        auto scaler = Number::load(scalers + j * Number::LIMBS);
        auto scaler_window = scaler.bit_slice(window_id * Config::s, Config::s);
        if (scaler_window != 0)
        {
          u32 old_count = atomicAdd(buckets_len.addr(window_id, scaler_window), 1);
          if (buckets_offset.get_const(window_id, scaler_window) + old_count >= len * Config::n_windows)
          {
            printf("Assertion failed! window_id: %d, scaler_window: %d, old_count: %d, offset: %d, len: %d\n",
                   window_id, scaler_window, old_count, buckets_offset.get_const(window_id, scaler_window), len);
            assert(false);
          }
          buckets_buffer[buckets_offset.get_const(window_id, scaler_window) + old_count] = PointId(window_id, j);
        }
      }
    }
  }

  // blockDim.x better be multiple of n_windows, and total number threads must be mutiple of n_windows
  template <typename Config>
  __global__ void initalize_sum(
      Array2D<Point, Config::n_buckets, Config::n_windows> sum)
  {
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    u32 window_id = tid % Config::n_windows;
    u32 i0_bucket = tid / Config::n_windows;

    u32 n_threads = gridDim.x * blockDim.x;
    u32 i_stride = n_threads / Config::n_windows;

    for (u32 i = i0_bucket; i < Config::n_buckets; i += i_stride)
    {
      sum.get(i, window_id) = Point::identity();
    }
  }

  // check sum is all intialized
  template <typename Config>
  __global__ void check_sum(Array2D<Point, Config::n_buckets, Config::n_windows> sum)
  {
    for (u32 i = 0; i < Config::n_buckets; i++)
    {
      for (u32 j = 0; j < Config::n_windows; j++)
      {
        if (!sum.get(i, j).x.is_zero() || sum.get(i, j).y != Element::one() || !sum.get(i, j).z.is_zero())
        {
          printf("Assertion failed! i: %d, j: %d\n",
                 i, j);
          assert(false);
        }
      }
    }
    printf("check true\n");
  }

  template <typename Config, u32 WarpPerBlock>
  __global__ void bucket_sum(
      const PointId *buckets_buffer,
      const Array2D<u32, Config::n_windows, Config::n_buckets> buckets_offset,
      const Array2D<u32, Config::n_windows, Config::n_buckets> buckets_len,
      const u32 *points,
      Array2D<Point, Config::n_buckets, Config::n_windows> sum)
  {
    using WarpReduce = cub::WarpReduce<Point>;
    __shared__ typename WarpReduce::TempStorage temp_storage[WarpPerBlock];

    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    u32 warp_id = tid / THREADS_PER_WARP;
    u32 in_warp_id = tid % THREADS_PER_WARP;

    u32 n_threads = gridDim.x * blockDim.x;
    u32 i_stride = n_threads / THREADS_PER_WARP; // number of warps
    u32 i0 = warp_id;

    // For each bucket...
    for (u32 i = i0; i < Config::n_buckets * Config::n_windows; i += i_stride)
    {
      u32 window_id = i % Config::n_windows;
      u32 bucket_no = i / Config::n_windows;

      if (bucket_no == 0 || buckets_len.get_const(window_id, bucket_no) == 0)
        continue;

      // A warp works independently to sum up points in the bucket
      auto acc = Point::identity();
      for (u32 j = in_warp_id; j < buckets_len.get_const(window_id, bucket_no); j += THREADS_PER_WARP)
      {
        auto p = PointAffine::load(
            points + buckets_buffer[buckets_offset.get_const(window_id, bucket_no) + j].scaler_id() * PointAffine::N_WORDS);
        acc = acc + p;
      }

      __syncwarp();

      // Sum up accumulated point from each thread in warp, returing the result to thread 0 in warp
      int in_block_warp_id = threadIdx.x / THREADS_PER_WARP;
      Point reduced_acc = WarpReduce(temp_storage[in_block_warp_id]).Reduce(acc, [](const Point &a, const Point &b)
                                                                            { return a + b; });

      if (in_warp_id == 0)
      {
        sum.get(bucket_no, window_id) = sum.get_const(bucket_no, window_id) + reduced_acc;
      }
    }
  }

  // A total of n_windows threads should be launched, only one block should be launched.
  // Opt. Opportunity: Perhaps this can be done faster by CPU; Or assign multiple threads to a window somehow
  template <typename Config>
  __global__ void bucket_reduction_and_window_reduction(
      const Array2D<Point, Config::n_buckets, Config::n_windows> buckets_sum,
      Point *reduced)
  {
    u32 window_id = threadIdx.x;

    auto acc = Point::identity();
    auto sum = Point::identity();

    // Bucket reduction
    // Perhaps save an iteration here by initializing acc to buckets_sum[n_buckets - 1][...]
    for (u32 i = Config::n_buckets - 1; i >= 1; i--)
    {
      acc = acc + buckets_sum.get_const(i, window_id);
      sum = sum + acc;
    }

    __syncthreads();

    // Window reduction
    // Opt. Opportunity: Diverges here
    for (u32 i = 0; i < window_id * Config::s; i++)
    {
      sum = sum.self_add();
    }

    __syncthreads();

    using BlockReduce = cub::BlockReduce<Point, Config::n_windows>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    Point msm_result = BlockReduce(temp_storage)
                           .Reduce(sum, [](const Point &a, const Point &b)
                                   { return a + b; }, blockDim.x);

    if (threadIdx.x == 0)
      *reduced = msm_result;
  }

  template <typename Config>
  __global__ void print_counts(const Array2D<u32, Config::n_windows, Config::n_buckets> counts)
  {
    printf("Buckets Count:\n");
    for (u32 i = 0; i < Config::n_windows; i++)
      for (u32 j = 0; j < Config::n_buckets; j++)
      {
        auto cnt = counts.get_const(i, j);
        if (cnt != 0)
          printf("Window %u, Bucket %x: %u\n", i, j, cnt);
      }
  }

  template <typename Config>
  __global__ void print_offsets(const Array2D<u32, Config::n_windows, Config::n_buckets> counts)
  {
    printf("Buckets Offset:\n");
    for (u32 i = 0; i < Config::n_windows; i++)
      for (u32 j = 0; j < Config::n_buckets; j++)
        printf("Window %u, Bucket %x: %u\n", i, j, counts.get_const(i, j));
  }

  template <typename Config>
  __global__ void print_lengths(const Array2D<u32, Config::n_windows, Config::n_buckets> counts)
  {
    printf("Buckets Length:\n");
    for (u32 i = 0; i < Config::n_windows; i++)
      for (u32 j = 0; j < Config::n_buckets; j++)
      {
        auto cnt = counts.get_const(i, j);
        if (cnt != 0)
          printf("Window %u, Bucket %x: %u\n", i, j, cnt);
      }
  }

  template <typename Config>
  void print_sums(const Array2D<Point, Config::n_buckets, Config::n_windows> sum)
  {
    Point *p = (Point *)malloc(Config::n_windows * Config::n_buckets * sizeof(Point));
    cudaMemcpy(p, sum.buf, Config::n_windows * Config::n_buckets * sizeof(Point), cudaMemcpyDeviceToHost);
    std::cout << "Buckets Sum:" << std::endl;
    for (u32 i = 0; i < Config::n_windows; i++)
      for (u32 j = 0; j < Config::n_buckets; j++)
      {
        auto point = p[i * Config::n_windows + j];
        if (!point.is_identity())
          std::cout << "Window " << i << ", Bucket " << j << point << std::endl;
      }
    free(p);
  }

  template <typename Config>
  __global__ void print_buckets(
      const PointId *buckets_buffer,
      const Array2D<u32, Config::n_windows, Config::n_buckets> buckets_offset,
      const Array2D<u32, Config::n_windows, Config::n_buckets> buckets_len)
  {
    printf("Buckets:\n");
    for (u32 i = 0; i < Config::n_windows; i++)
      for (u32 j = 0; j < Config::n_buckets; j++)
      {
        auto len = buckets_len.get_const(i, j);
        // if (len > 0)
        // {
          printf("Window %u, Bucket %x: ", i, j);
          for (u32 k = 0; k < len; k++)
          {
            PointId id = buckets_buffer[buckets_offset.get_const(i, j) + k];
            printf("%u ", id.scaler_id());
          }
          printf("\n");
        // }
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

    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Count items in buckets
    u32 *scalers;
    PROPAGATE_CUDA_ERROR(cudaMalloc(&scalers, sizeof(u32) * Number::LIMBS * len));
    u32 *counts_buf;
    PROPAGATE_CUDA_ERROR(cudaMalloc(&counts_buf, sizeof(u32) * Config::n_windows * Config::n_buckets));
    PROPAGATE_CUDA_ERROR(cudaMemcpy(scalers, h_scalers, sizeof(u32) * Number::LIMBS * len, cudaMemcpyHostToDevice));
    auto counts = Array2D<u32, Config::n_windows, Config::n_buckets>(counts_buf);

    u32 block_size = 96;
    u32 grid_size = 128;
    initialize_counts<Config><<<1, block_size>>>(counts);
    count_buckets<Config><<<grid_size, block_size>>>(scalers, len, counts);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
      std::cerr << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: "
                << cudaGetErrorString(err) << " (Error Code: " << err << ")" << std::endl;
    }

    if (Config::debug)
    {
      print_counts<Config><<<1, 1>>>(counts);
    }

    // Allocate space for buckets buffer
    // Space for PoindId's
    PointId *buckets_buffer;
    PROPAGATE_CUDA_ERROR(cudaMalloc(&buckets_buffer, sizeof(PointId) * len * Config::n_windows));

    // Initialize bucket offsets
    u32 *buckets_offset_buf;
    PROPAGATE_CUDA_ERROR(cudaMalloc(&buckets_offset_buf, sizeof(u32) * Config::n_windows * Config::n_buckets));
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        counts_buf, buckets_offset_buf, Config::n_buckets * Config::n_buckets);
    // Allocate temporary storage
    PROPAGATE_CUDA_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        counts_buf, buckets_offset_buf, Config::n_buckets * Config::n_buckets);
    PROPAGATE_CUDA_ERROR(cudaFree(d_temp_storage));
    auto buckets_offset = Array2D<u32, Config::n_windows, Config::n_buckets>(buckets_offset_buf);

    if (Config::debug)
    {
      print_offsets<Config><<<1, 1>>>(buckets_offset);
    }

    // Scatter
    block_size = 96;
    grid_size = 128;
    initialize_buckets_len<Config><<<1, block_size>>>(counts);
    scatter<Config><<<grid_size, block_size>>>(
        scalers,
        len,
        buckets_buffer,
        buckets_offset,
        counts);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
      std::cerr << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: "
                << cudaGetErrorString(err) << " (Error Code: " << err << ")" << std::endl;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "MSM scatter time:" << elapsedTime << std::endl;
    // space for counts is reused
    PROPAGATE_CUDA_ERROR(cudaFree(scalers));
    // print_buckets<Config><<<1, 1>>>(buckets_buffer, buckets_offset, counts);

    if (Config::debug)
    {
      // print_lengths<Config><<<1, 1>>>(counts);
      print_buckets<Config><<<1, 1>>>(buckets_buffer, buckets_offset, counts);
    }

    elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Prepare for bucket sum
    Point *buckets_sum_buf;
    PROPAGATE_CUDA_ERROR(cudaMalloc(&buckets_sum_buf, sizeof(Point) * Config::n_windows * Config::n_buckets));
    auto buckets_sum = Array2D<Point, Config::n_buckets, Config::n_windows>(buckets_sum_buf);

    block_size = 96;
    grid_size = 128;
    initalize_sum<Config><<<grid_size, block_size>>>(
        buckets_sum);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
      std::cerr << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: "
                << cudaGetErrorString(err) << " (Error Code: " << err << ")" << std::endl;
    }
    // check_sum<Config><<<1, 1>>>(buckets_sum);

    // TODO: Cut into chunks to support len at 2^30
    u32 *points;
    PROPAGATE_CUDA_ERROR(cudaMalloc(&points, sizeof(u32) * PointAffine::N_WORDS * len));
    PROPAGATE_CUDA_ERROR(cudaMemcpy(points, h_points, sizeof(u32) * PointAffine::N_WORDS * len, cudaMemcpyHostToDevice));

    // Do bucket sum
    block_size = 8 * THREADS_PER_WARP;
    grid_size = 256;
    bucket_sum<Config, 2><<<grid_size, block_size>>>(
        buckets_buffer,
        buckets_offset,
        counts,
        points,
        buckets_sum);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "MSM bucket sum time:" << elapsedTime << std::endl;

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
      std::cerr << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: "
                << cudaGetErrorString(err) << " (Error Code: " << err << ")" << std::endl;
    }

    if (Config::debug)
    {
      print_sums<Config>(buckets_sum);
    }

    elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Bucket reduction and window reduction
    Point *reduced;
    PROPAGATE_CUDA_ERROR(cudaMalloc(&reduced, sizeof(Point)));

    block_size = Config::n_windows;
    grid_size = 1;
    bucket_reduction_and_window_reduction<Config><<<grid_size, block_size>>>(
        buckets_sum,
        reduced);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
      std::cerr << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: "
                << cudaGetErrorString(err) << " (Error Code: " << err << ")" << std::endl;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "MSM bucket reduce time:" << elapsedTime << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    PROPAGATE_CUDA_ERROR(cudaMemcpy(&h_result, reduced, sizeof(Point), cudaMemcpyDeviceToHost));

    return cudaSuccess;
  }

}
