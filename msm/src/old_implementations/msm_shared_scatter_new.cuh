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

  template <typename T, u32 D1, u32 D2, u32 D3>
  struct Array3D
  {
    T *buf;

    Array3D(T *buf) : buf(buf) {}
    __host__ __device__ __forceinline__ T &get(u32 i, u32 j, u32 k)
    {
      return buf[i * D2 * D3 + j * D3 + k];
    }
    __host__ __device__ __forceinline__ const T &get_const(u32 i, u32 j, u32 k) const
    {
      return buf[i * D2 * D3 + j * D3 + k];
    }
    __host__ __device__ __forceinline__ T *addr(u32 i, u32 j, u32 k)
    {
      return buf + i * D2 * D3 + j * D3 + k;
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

    static constexpr u32 grid_size = 32768;
    static constexpr u32 block_size = 64;
  };

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
    //     if(counts.get(i, j, blockIdx.x) != 0)
    //     {
    //       printf("Assertion failed! i: %d, j: %d, value: %d\n",
    //             i, j, counts.get(i, j, blockIdx.x));
    //        assert(false);
    //     }
    //   }
    // }
  }

  // with n_windows = 32, launched 64 threads
  //
  //   t0 -> scaler 0 = digit 0 | d1 | ... | d31
  //
  //   t1 -> scaler 1 = digit 0 | d1 | ... | d31
  //
  //   tn -> scaler n = digit 0 | d1 | ... | d31
  //
  //   t0 -> scaler n+1 = digit 0 | d1 | ... | d31
  //   ...
  //
  // blockDim.x better be multiple of n_windows, and total number threads must be mutiple of n_windows
  template <typename Config>
  __global__ void count_buckets(
      const u32 *scalers,
      const u32 len,
      PointId* buckets_buffer,
      Array2D<u32, Config::n_windows, Config::n_buckets> buckets_len)
  {
    u32 i0_scaler = blockDim.x * blockIdx.x + threadIdx.x;

    u32 n_threads = gridDim.x * blockDim.x;
    u32 i_stride = n_threads;

    __shared__ u32 shm_counts[Config::n_windows][Config::n_buckets];
    __shared__ u32 shm_cache[2048];

    if (threadIdx.x < Config::n_windows)
    {
      u32 window_id = threadIdx.x;
      for (u32 i = 0; i < Config::n_buckets; i++)
      {
        shm_counts[window_id][i] = 0;
      }
    }
    
    __syncthreads();

    // Count into block-wide counter
    for (u32 i = i0_scaler; i < len && blockIdx.x < Config::grid_size; i += i_stride)
    {
      auto scaler = Number::load(scalers + i * Number::LIMBS);
      for(u32 j = 0; j < Config::n_windows; j++)
      {
        auto scaler_window = scaler.bit_slice(j * Config::s, Config::s);
        if (scaler_window != 0)
          atomicAdd(&shm_counts[j][scaler_window], 1);
      }        
    }

    __syncthreads();

    // if(blockIdx.x == 0 && threadIdx.x == 0)
    //   printf("shm_counts:\n");

    // for (u32 i = 0; i < Config::n_windows && blockIdx.x == 0 && threadIdx.x == 0; i++)
    //   for (u32 j = 0; j < Config::n_buckets; j++)
    //   {
    //     auto cnt = shm_counts[i][j];
    //     printf("Window %u, Bucket %x, Block %x: %u\n", i, j, blockIdx.x, cnt);
    //   }

    // __syncthreads();
    
    u32 local_lens[Config::n_buckets];

    if (threadIdx.x < Config::n_windows)
    {
      for(int i=0;i<Config::n_buckets;++i)
        local_lens[i] = shm_counts[threadIdx.x][i];
    }
    else
    {
      for(int i=0;i<Config::n_buckets;++i)
        local_lens[i] = 0;
    }

    __syncthreads();

    // 创建BlockScan实例
    using BlockScan = cub::BlockScan<u32, 128>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    // 使用ExclusiveSum计算每个桶的前缀和
    BlockScan(temp_storage).ExclusiveSum(local_lens, local_lens);

    __syncthreads();
    
    // if(blockIdx.x == 0 && threadIdx.x < 5)
    // {
    //   for(int i=0;i<Config::n_buckets;++i)
    //     printf("window:%d bucket:%d offset:%d\n",threadIdx.x,i,local_lens[i]);
    // }

    // __syncthreads();

    // 将前缀和结果写回共享内存
    if (threadIdx.x < Config::n_windows)
    {
      for(int i=0;i<Config::n_buckets;++i)
      {
        shm_counts[threadIdx.x][i] = local_lens[i];
      }  
    }

    __syncthreads();

    // if(blockIdx.x == 0 && threadIdx.x == 0)
    //   printf("shm_offsets:\n");
    // for(int i=0;blockIdx.x == 0 && threadIdx.x == 0 && i<Config::n_windows;++i)
    // {
    //   for(int j=0;j<Config::n_buckets;++j)
    //   {
    //     printf("Window:%d Bucket:%d offset:%d\n",i,j,shm_counts[i][j]);
    //   }
    // }

    // __syncthreads();

    // Opt. Opportunity: First scatter to shared memory, then scatter to global, so as to reduce
    // global atomic operations.
    for (u32 i = i0_scaler; i < len; i += i_stride)
    {
      auto scaler = Number::load(scalers + i * Number::LIMBS);
      for(u32 j = 0; j < Config::n_windows; j++)
      {
        auto scaler_window = scaler.bit_slice(j * Config::s, Config::s);
        if (scaler_window != 0)
        {
          u32 old_count = atomicAdd(&shm_counts[j][scaler_window], 1);
          if (old_count >= 2048)
          {
            printf("Assertion failed! window_id: %d, scaler_window: %d, old_count: %d\n",
                   j, scaler_window, old_count);
            assert(false);
          }
          shm_cache[old_count] = i;
        }
      }        
    }

    __syncthreads();

    // if(blockIdx.x == 0 && threadIdx.x == 0)
    //   printf("after_shm_offsets:\n");
    // for(int i=0;blockIdx.x == 0 && threadIdx.x == 0 && i<Config::n_windows;++i)
    // {
    //   for(int j=0;j<Config::n_buckets;++j)
    //   {
    //     printf("Window:%d Bucket:%d offset:%d\n",i,j,shm_counts[i][j]);
    //   }
    // }

    // if(blockIdx.x == 0 && threadIdx.x == 0)
    // {
    //   printf("shm cache:");
    //   for(int i = 0; i < shm_counts[Config::n_windows-1][Config::n_buckets-1]; ++i)
    //   {
    //     printf("%d ", shm_cache[i]);
    //   }
    // }

    // __syncthreads();

    i_stride = Config::block_size;
    for(u32 i = threadIdx.x; i < Config::n_windows * Config::n_buckets; i += i_stride)
    {
      int window_id = i / Config::n_buckets;
      int bucket_id = i % Config::n_buckets;

      u32 bucket_len = 0;
      u32 in_block_offset_0 = 0;
      if (window_id == 0 && bucket_id == 0)
      {
        bucket_len = shm_counts[window_id][bucket_id];
      }  
      else if(bucket_id == 0)
      {
        bucket_len = shm_counts[window_id][bucket_id] - shm_counts[window_id-1][Config::n_buckets-1];
        in_block_offset_0 = shm_counts[window_id-1][Config::n_buckets-1];
      }  
      else
      {
        bucket_len = shm_counts[window_id][bucket_id] - shm_counts[window_id][bucket_id-1];
        in_block_offset_0 = shm_counts[window_id][bucket_id-1];
      }  
    //   // if(blockIdx.x == 0)
    //   //   printf("Window: %d, Bucket: %d, len: %d\n",window_id,bucket_id,len);
      u32 old_count = atomicAdd(&buckets_len.get(window_id, bucket_id), bucket_len);
      assert(old_count < len / 16);
      if(old_count >= len / 16)
      {
        printf("Window: %d Bucket: %x count: %d\n",window_id,bucket_id,old_count);
        assert(old_count < len / 16);
      }
      u32 buckets_buffer_offset_0 = len / 16 * i + old_count;     
      
      for(int j=0; j < bucket_len; ++j)
      { 
        assert(shm_cache[in_block_offset_0 + j] != 1073741824);       
        buckets_buffer[buckets_buffer_offset_0 + j] = PointId(window_id, shm_cache[in_block_offset_0 + j]);
        shm_cache[in_block_offset_0 + j] = 1073741824;
      }
    }

    __syncthreads();

    // if(blockIdx.x == 0 && threadIdx.x == 0)
    // {
    //   printf("\nbuckets_len:\n");
    //   for(int i=0;i<Config::n_windows;++i)
    //   {
    //     for(int j=0;j<Config::n_buckets;++j)
    //     {
    //       printf("Window:%d, Bucket:%x, len:%d\n",i,j,buckets_len.get_const(i,j));
    //     }
    //   }
    // }

    // if(blockIdx.x == 0 && threadIdx.x == 0)
    // {
    //   printf("\nbuckets_buffer:\n");
    //   for(int i=0;i<Config::n_windows;++i)
    //   {
    //     for(int j=0;j<Config::n_buckets;++j)
    //     {
    //       printf("Window %d, Bucket %x: ",i,j);
    //       int offset = len/16*(i*Config::n_buckets+j);
    //       for(int k=0;k<buckets_len.get_const(i,j);++k)
    //         printf("%d ",buckets_buffer[offset+k].scaler_id());
    //       printf("\n");
    //     }
    //   }
    // }
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
      u32 len,
      const PointId *buckets_buffer,
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

      if (bucket_no == 0)
        continue;

      // A warp works independently to sum up points in the bucket
      auto acc = Point::identity();
      u32 bucket_len = buckets_len.get_const(window_id, bucket_no);
      for (u32 j = in_warp_id; j < bucket_len; j += THREADS_PER_WARP)
      {
        auto p = PointAffine::load(
            points + buckets_buffer[len / 16 * (window_id * Config::n_buckets + bucket_no) + j].scaler_id() * PointAffine::N_WORDS);
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
  __global__ void print_counts(const Array3D<u32, Config::n_windows, Config::n_buckets, Config::grid_size> counts)
  {
    printf("Buckets Count:\n");
    for (u32 i = 0; i < Config::n_windows; i++)
      for (u32 j = 0; j < Config::n_buckets; j++)
        for (u32 k = 0; k < Config::grid_size; k++)
        {
          auto cnt = counts.get_const(i, j, k);
          printf("Window %u, Bucket %x, Block %x: %u\n", i, j, k, cnt);
        }
  }

  template <typename Config>
  __global__ void print_counts_buf(const u32* counts_buf)
  {
    printf("counts_buf size: %ld\n", sizeof(counts_buf)/sizeof(u32));
    for(int i=25;i<Config::n_windows-1;++i)
    {
      for(int j=0;j<Config::n_buckets;++j)
      {
        for(int k=0;k<Config::grid_size;++k)
        {
          printf("Window %u, Bucket %x, Block %x: %u\n", i, j, k, counts_buf[i*Config::n_buckets*Config::grid_size+j*Config::grid_size+k]);
        }
      }
    }
  }

  template <typename Config>
  __global__ void print_offsets(const Array3D<u32, Config::n_windows, Config::n_buckets, Config::grid_size> counts)
  {
    printf("Buckets Offset:\n");
    for (u32 i = 0; i < Config::n_windows; i++)
      for (u32 j = 0; j < Config::n_buckets; j++)
        printf("Window %u, Bucket %x: %u\n", i, j, counts.get_const(i, j, 0));
    // for(u32 j = 0; j < Config::n_buckets; j++)
      // for(u32 k = 0; k < Config::grid_size; k++)
      //   printf("Window 0, Bucket 1, Block %x: %u\n", k, counts.get_const(0, 1, k));
  }

  template <typename Config>
  __global__ void print_offsets_buf(const u32* offsets_buf)
  {
    for(int i=24;i<26;++i)
    {
      for(int j=0;j<Config::n_buckets;++j)
      {
        for(int k=0;k<Config::grid_size;++k)
        {
          printf("Window %u, Bucket %x, Block %x: %u\n", i, j, k, offsets_buf[i*Config::n_buckets*Config::grid_size+j*Config::grid_size+k]);
        }
      }
    }
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
      u32 len,
      const PointId *buckets_buffer,
      const Array2D<u32, Config::n_windows, Config::n_buckets> buckets_len)
  {
    for (u32 i = 0; i < Config::n_windows; i++)
      for (u32 j = 0; j < Config::n_buckets; j++)
      {
        printf("Window %u, Bucket %x: ", i, j);
        u32 start = len/16*(i*Config::n_buckets+j);
        for (u32 k = start; k < start+buckets_len.get_const(i,j); k++)
        {
          PointId id = buckets_buffer[k];
          printf("%u ", id.scaler_id());
        }
        printf("\n");
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
    PROPAGATE_CUDA_ERROR(cudaMemcpy(scalers, h_scalers, sizeof(u32) * Number::LIMBS * len, cudaMemcpyHostToDevice));
    
    // Allocate space for buckets buffer
    // Space for PoindId's
    PointId *buckets_buffer;
    PROPAGATE_CUDA_ERROR(cudaMalloc(&buckets_buffer, sizeof(PointId) * len / 16 * Config::n_windows * Config::n_buckets));
    u32 *buckets_len_buf;
    PROPAGATE_CUDA_ERROR(cudaMalloc(&buckets_len_buf, sizeof(u32) * Config::n_windows * Config::n_buckets));
    auto buckets_len = Array2D<u32, Config::n_windows, Config::n_buckets>(buckets_len_buf);

    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    initialize_counts<Config><<<Config::grid_size, Config::n_windows>>>(buckets_len);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
      std::cerr << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: "
                << cudaGetErrorString(err) << " (Error Code: " << err << ")" << std::endl;
    }
    count_buckets<Config><<<Config::grid_size, Config::block_size>>>(scalers, len, buckets_buffer, buckets_len);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      std::cerr << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: "
                << cudaGetErrorString(err) << " (Error Code: " << err << ")" << std::endl;
    }

    // print_buckets<Config><<<1, 1>>>(len, buckets_buffer, buckets_len);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "MSM scatter time:" << elapsedTime << std::endl;   
        
    PROPAGATE_CUDA_ERROR(cudaFree(scalers));


    // Prepare for bucket sum
    Point *buckets_sum_buf;
    PROPAGATE_CUDA_ERROR(cudaMalloc(&buckets_sum_buf, sizeof(Point) * Config::n_windows * Config::n_buckets));
    auto buckets_sum = Array2D<Point, Config::n_buckets, Config::n_windows>(buckets_sum_buf);

    // TODO: Cut into chunks to support len at 2^30
    u32 *points;
    PROPAGATE_CUDA_ERROR(cudaMalloc(&points, sizeof(u32) * PointAffine::N_WORDS * len));
    PROPAGATE_CUDA_ERROR(cudaMemcpy(points, h_points, sizeof(u32) * PointAffine::N_WORDS * len, cudaMemcpyHostToDevice));

    elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    initalize_sum<Config><<<Config::grid_size, Config::block_size>>>(
        buckets_sum);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
      std::cerr << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: "
                << cudaGetErrorString(err) << " (Error Code: " << err << ")" << std::endl;
    }
    // check_sum<Config><<<1, 1>>>(buckets_sum);

    // Do bucket sum
    u32 block_size = 8 * THREADS_PER_WARP;
    u32 grid_size = 256;
    bucket_sum<Config, 8><<<grid_size, block_size>>>(
        len,
        buckets_buffer,
        buckets_len,
        points,
        buckets_sum);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      std::cerr << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: "
                << cudaGetErrorString(err) << " (Error Code: " << err << ")" << std::endl;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "MSM sum time:" << elapsedTime << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    PROPAGATE_CUDA_ERROR(cudaFree(points));
    PROPAGATE_CUDA_ERROR(cudaFree(buckets_buffer));
    PROPAGATE_CUDA_ERROR(cudaFree(buckets_len_buf));

    if (Config::debug)
    {
      print_sums<Config>(buckets_sum);
    }

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
    PROPAGATE_CUDA_ERROR(cudaFree(buckets_sum_buf));

    PROPAGATE_CUDA_ERROR(cudaMemcpy(&h_result, reduced, sizeof(Point), cudaMemcpyDeviceToHost));
    PROPAGATE_CUDA_ERROR(cudaFree(reduced));

    return cudaSuccess;
  }

}
