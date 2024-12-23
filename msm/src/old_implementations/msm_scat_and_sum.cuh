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

    static constexpr u32 grid_size = 256;
    static constexpr u32 block_size = 128;
  };

  __global__ void upsweep(u32* offsets, int n, int step) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = step * 2;
    
    if (idx < n / offset) {
        int i = idx * offset + step - 1;
        int j = i + step;
        if (j < n) {
            offsets[j] += offsets[i];
        }
    }
}

__global__ void downsweep(u32* offsets, int n, int step) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = step * 2;
    
    if (idx < n / offset) {
        int i = idx * offset + step - 1;
        int j = i + step;
        if (j < n) {
            int temp = offsets[i];
            offsets[i] = offsets[j];
            offsets[j] += temp;
        }
    }
}

  // blockDim.x better be multiple of n_windows, and total number threads must be mutiple of n_windows
  // template <typename Config>
  // __global__ void initalize_sum(
  //     Array3D<Point, Config::n_buckets, Config::n_windows, Config::grid_size> sum)
  // {
  //   u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  //   u32 i0_bucket = tid / (Config::n_windows * Config::grid_size);
  //   u32 window_id = tid / Config::grid_size;
  //   u32 block_id = tid % Config::grid_size;
    
  //   u32 i_stride = gridDim.x * blockDim.x / (Config::n_windows * Config::grid_size);

  //   for (u32 i = i0_bucket; i < Config::n_buckets; i += i_stride)
  //   {
  //     sum.get(i, window_id, block_id) = Point::identity();
  //   }

  // }

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

  template <typename Config>
  __global__ void in_block_sum(
      const u32 len,
      const u32 *scalers,
      const u32 *points,
      Array3D<Point, Config::n_buckets, Config::n_windows, Config::grid_size> sum)
  {
    using BlockReduce = cub::BlockReduce<Point, Config::block_size>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    u32 blocksPerWindow = Config::grid_size / Config::n_windows;
    u32 i0_scaler = threadIdx.x + (blockIdx.x % blocksPerWindow) * blockDim.x;
    u32 window_id = blockIdx.x / blocksPerWindow;

    u32 i_stride = blocksPerWindow * blockDim.x;

    Point acc[Config::n_buckets];

    for(u32 i = 0; i < Config::n_buckets; ++i)
    {
      acc[i] = Point::identity();
    }  

    // Opt. Opportunity: First scatter to shared memory, then scatter to global, so as to reduce
    // global atomic operations.
    for (u32 i = i0_scaler; i < len; i += i_stride)
    {
      auto scaler = Number::load(scalers + i * Number::LIMBS);      
      auto scaler_window = scaler.bit_slice(window_id * Config::s, Config::s);
      if (scaler_window != 0)
      { 
        auto p = PointAffine::load(points + i * PointAffine::N_WORDS);        
        acc[scaler_window] = acc[scaler_window] + p;
      }
    }
    
    __syncthreads();

    // if(threadIdx.x == 0)
    // {
    //   for(u32 i = 0; i < Config::n_buckets; ++i)
    //   {
    //     printf("i %u, j %u, acc 0x%x_%x_%x_%x_%x_%x_%x_%x, 0x%x_%x_%x_%x_%x_%x_%x_%x, 0x%x_%x_%x_%x_%x_%x_%x_%x\n", i, window_id,
    //          acc[i].x.n.limbs[7], acc[i].x.n.limbs[6], acc[i].x.n.limbs[5], acc[i].x.n.limbs[4], acc[i].x.n.limbs[3], acc[i].x.n.limbs[2], acc[i].x.n.limbs[1], acc[i].x.n.limbs[0],
    //          acc[i].y.n.limbs[7], acc[i].y.n.limbs[6], acc[i].y.n.limbs[5], acc[i].y.n.limbs[4], acc[i].y.n.limbs[3], acc[i].y.n.limbs[2], acc[i].y.n.limbs[1], acc[i].y.n.limbs[0],
    //          acc[i].z.n.limbs[7], acc[i].z.n.limbs[6], acc[i].z.n.limbs[5], acc[i].z.n.limbs[4], acc[i].z.n.limbs[3], acc[i].z.n.limbs[2], acc[i].z.n.limbs[1], acc[i].z.n.limbs[0]);
    //   }  
    // } 
   
    __shared__ Point reduced_acc;

    for(u32 i = 0; i < Config::n_buckets; ++i)
      {
        reduced_acc = BlockReduce(temp_storage).Reduce(acc[i], [](const Point &a, const Point &b)
                                                                          { return a + b; });
        if(threadIdx.x == 0)
          sum.get(i, window_id, blockIdx.x) = reduced_acc;
      }

    // u32 i0_scaler = blockDim.x * blockIdx.x + threadIdx.x;

    // u32 n_threads = gridDim.x * blockDim.x;
    // u32 i_stride = n_threads;

    // __shared__ Point shm_sum[Config::n_buckets][Config::n_windows];
    // __shared__ char shm_lock[Config::n_buckets][Config::n_windows];

    // if(threadIdx.x < Config::n_buckets * Config::n_windows)
    // {
    //   u32 bucket_id = threadIdx.x / Config::n_windows;
    //   u32 window_id = threadIdx.x % Config::n_windows;
    //   shm_sum[bucket_id][window_id] = Point::identity();
    //   shm_lock[bucket_id][window_id] = 0;
    // }

    // __syncthreads();

    // // Opt. Opportunity: First scatter to shared memory, then scatter to global, so as to reduce
    // // global atomic operations.
    // for (u32 i = i0_scaler; i < len; i += i_stride)
    // {
    //   auto scaler = Number::load(scalers + i * Number::LIMBS);
    //   for(u32 j = 0; j < Config::n_windows; j++)
    //   {
    //     auto scaler_window = scaler.bit_slice(j * Config::s, Config::s);
    //     if (scaler_window != 0)
    //     {
    //       Point old = shm_sum[scaler_window][j], expect;
    //       do
    //       {
    //         expect = old;
    //         old = atomicCAS(&shm_sum[scaler_window][j], expect, expect + PointAffine::load(points + i * PointAffine::N_WORDS));
    //       }
    //       while(!(expect == old));
    //     }
    //   }        
    // }

    // __syncthreads();

    // if(threadIdx.x < Config::n_buckets * Config::n_windows)
    // {
    //   u32 bucket_id = threadIdx.x / Config::n_windows;
    //   u32 window_id = threadIdx.x % Config::n_windows;
    //   Point old = sum[bucket_id][window_id], expect;
    //   do
    //   {
    //     expect = old;
    //     old = atomicCAS(&sum[bucket_id][window_id], expect, expect + shm_sum[bucket_id][window_id]);
    //   }
    //   while(!(expect == old));
    // }
  }

  template <typename Config>
  __global__ void block_sum(
      Array3D<Point, Config::n_buckets, Config::n_windows, Config::grid_size> sum,
      Array2D<Point, Config::n_buckets, Config::n_windows> result_sum)
  {
    using BlockReduce = cub::BlockReduce<Point, Config::block_size>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    u32 i0 = blockIdx.x;

    u32 i0_scaler = i0 / Config::n_windows;
    u32 i0_window = i0 % Config::n_windows;

    auto acc = sum.get_const(i0_scaler, i0_window, threadIdx.x);

    __syncthreads();

    Point reduced_acc = BlockReduce(temp_storage).Reduce(acc, [](const Point &a, const Point &b)
                                                                            { return a + b; });

    if (threadIdx.x == 0)
    {
      result_sum.get(i0_scaler, i0_window) = reduced_acc;
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
  void print_3D_sums(const Array3D<Point, Config::n_buckets, Config::n_windows, Config::grid_size> sum)
  {
    Point *p = (Point *)malloc(Config::n_windows * Config::n_buckets * Config::grid_size * sizeof(Point));
    cudaMemcpy(p, sum.buf, Config::n_windows * Config::n_buckets * Config::grid_size * sizeof(Point), cudaMemcpyDeviceToHost);
    std::cout << "3D Buckets Sum:" << std::endl;
    for (u32 i = 0; i < Config::n_windows; i++)
      for (u32 j = 0; j < Config::n_buckets; j++)
        for (u32 k = 0; k < Config::grid_size; k++)
        {
          auto point = p[i * Config::n_windows * Config::grid_size + j * Config::grid_size + k];
          if (!point.is_identity())
            std::cout << "Window " << i << ", Bucket " << j << ", Block " << k << point << std::endl;
        }
    free(p);
  }

  // template <typename Config>
  // __global__ void print_buckets(
  //     const PointId *buckets_buffer,
  //     const Array3D<u32, Config::n_windows, Config::n_buckets, Config::grid_size> buckets_offset,
  //     const Array3D<u32, Config::n_windows, Config::n_buckets, Config::grid_size> buckets_len)
  // {
  //   printf("Buckets:\n");
  //   for (u32 i = 0; i < Config::n_windows; i++)
  //     for (u32 j = 0; j < Config::n_buckets; j++)
  //     {
  //       printf("Window %u, Bucket %x: ", i, j);
  //       if(j + 1 < Config::n_buckets)
  //       {
  //         for (u32 k = buckets_offset.get_const(i, j, 0); k < buckets_offset.get_const(i, j+1, 0); k++)
  //         {
  //           PointId id = buckets_buffer[k];
  //           printf("%u ", id.scaler_id());
  //         }
  //       }
  //       else if(i + 1 < Config::n_windows)
  //       {
  //         for (u32 k = buckets_offset.get_const(i, j, 0); k < buckets_offset.get_const(i+1, 0, 0); k++)
  //         {
  //           PointId id = buckets_buffer[k];
  //           printf("%u ", id.scaler_id());
  //         }
  //       }
  //       printf("\n");
  //     }
  // }

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
    

    // initalize_sum<Config><<<Config::grid_size, Config::block_size>>>(
    //     block_sums);
    // err = cudaDeviceSynchronize();
    // if (err != cudaSuccess)
    // {
    //   std::cerr << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: "
    //             << cudaGetErrorString(err) << " (Error Code: " << err << ")" << std::endl;
    // }
    // check_sum<Config><<<1, 1>>>(buckets_sum);

    // TODO: Cut into chunks to support len at 2^30
    u32 *scalers;
    PROPAGATE_CUDA_ERROR(cudaMalloc(&scalers, sizeof(u32) * Number::LIMBS * len));
    PROPAGATE_CUDA_ERROR(cudaMemcpy(scalers, h_scalers, sizeof(u32) * Number::LIMBS * len, cudaMemcpyHostToDevice));
    u32 *points;
    PROPAGATE_CUDA_ERROR(cudaMalloc(&points, sizeof(u32) * PointAffine::N_WORDS * len));
    PROPAGATE_CUDA_ERROR(cudaMemcpy(points, h_points, sizeof(u32) * PointAffine::N_WORDS * len, cudaMemcpyHostToDevice));

    Point *in_block_sums_buf;
    PROPAGATE_CUDA_ERROR(cudaMalloc(&in_block_sums_buf, sizeof(Point) * Config::n_windows * Config::n_buckets * Config::grid_size));
    auto in_block_sums = Array3D<Point, Config::n_buckets, Config::n_windows, Config::grid_size>(in_block_sums_buf);

    assert(scalers != nullptr && points != nullptr && in_block_sums_buf != nullptr);
    // Do bucket sum
    in_block_sum<Config><<<Config::grid_size, Config::block_size>>>(
        len,
        scalers,
        points,
        in_block_sums);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
      std::cerr << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: "
                << cudaGetErrorString(err) << " (Error Code: " << err << ")" << std::endl;
    }

    // print_3D_sums<Config>(in_block_sums);

    // if (Config::debug)
    // {
    //   print_sums<Config>(in_block_sums);
    // }

    Point *block_sums_buf;
    PROPAGATE_CUDA_ERROR(cudaMalloc(&block_sums_buf, sizeof(Point) * Config::n_windows * Config::n_buckets));
    auto block_sums = Array2D<Point, Config::n_buckets, Config::n_windows>(block_sums_buf);

    block_sum<Config><<<Config::n_windows * Config::n_buckets, Config::grid_size>>>(
        in_block_sums,
        block_sums);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
      std::cerr << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: "
                << cudaGetErrorString(err) << " (Error Code: " << err << ")" << std::endl;
    }

    print_sums<Config>(block_sums);

    // Bucket reduction and window reduction
    Point *reduced;
    PROPAGATE_CUDA_ERROR(cudaMalloc(&reduced, sizeof(Point)));

    u32 block_size = Config::n_windows;
    u32 grid_size = 1;
    bucket_reduction_and_window_reduction<Config><<<grid_size, block_size>>>(
        block_sums,
        reduced);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
      std::cerr << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: "
                << cudaGetErrorString(err) << " (Error Code: " << err << ")" << std::endl;
    }

    PROPAGATE_CUDA_ERROR(cudaMemcpy(&h_result, reduced, sizeof(Point), cudaMemcpyDeviceToHost));

    return cudaSuccess;
  }

}
