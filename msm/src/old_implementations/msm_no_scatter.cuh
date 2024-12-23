#include "../../mont/src/bn254_scalar.cuh"
#include "bn254.cuh"

#include <cub/cub.cuh>
#include <iostream>
#include <cuda/std/tuple>
#include <cuda/std/iterator>

#define PROPAGATE_CUDA_ERROR(x)                                                                                    \
  {                                                                                                                \
    err = x;                                                                                                       \
    if (err != cudaSuccess)                                                                                        \
    {                                                                                                              \
      std::cerr << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: " << cudaGetErrorString(err) << std::endl; \
      return err;                                                                                                  \
    }                                                                                                              \
  }
// #define DEBUG
namespace msm
{
    using bn254::Point;
    using bn254::PointAffine;
    using bn254_scalar::Element;
    using bn254_scalar::Number;
    using mont::u32;
    using mont::u64;
    using mont::usize;

    const u32 THREADS_PER_WARP = 32;

    constexpr u32 pow2(u32 n)
    {
        return n == 0 ? 1 : 2 * pow2(n - 1);
    }

    constexpr __device__ __host__ u32 div_ceil(u32 a, u32 b)
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

        __host__ __device__ Array2D(T *buf) : buf(buf) {}
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
        static constexpr u32 block_dim = 256;

        static constexpr bool debug = true;
    };

   template <typename Config>
    __global__ void direct_bucket_sum(
        const u32 len, // number of scalars
        u32 *scalers, // all scalers
        u32 *points, // all points
        Array2D<Point, Config::n_windows, Config::n_buckets - 1> sum, // sum of each bucket
        Array2D<unsigned short int, Config::n_windows, Config::n_buckets - 1> mutex // mutex for each bucket
    ) {

        const u32 gtid = blockIdx.x * blockDim.x + threadIdx.x;
        const u32 iter_per_thread = div_ceil(len, gridDim.x * blockDim.x);
        const u32 start_pos = gtid * iter_per_thread;
        const u32 end_pos = min(start_pos + iter_per_thread, len);

        for (u64 i = start_pos; i < end_pos; i++) {
            auto scaler = Number::load(scalers + i * Number::LIMBS);
            auto point = PointAffine::load(points + i * PointAffine::N_WORDS);
            for (u32 j = 0; j < Config::n_windows; j++) {
                auto bucket_scalar = scaler.bit_slice(j * Config::s, Config::s);
                if (bucket_scalar != 0) {
                    auto mutex_ptr = mutex.addr(j, bucket_scalar - 1);
                    u32 time = 1;
                    while (atomicCAS(mutex_ptr, (unsigned short int)0, (unsigned short int)1) != 0) {
                        __nanosleep(time);
                        if (time < 64) time *= 2;
                    }
                    __threadfence();
                    sum.get(j, bucket_scalar - 1) = sum.get(j, bucket_scalar - 1) + point;
                    __threadfence();
                    atomicCAS(mutex_ptr, (unsigned short int)1, (unsigned short int)0);
                }
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
        u32 grid = std::min(deviceProp.multiProcessorCount, (int)div_ceil(len, Config::block_dim));

        u32 *scalers;
        PROPAGATE_CUDA_ERROR(cudaMalloc(&scalers, sizeof(u32) * Number::LIMBS * len));
        PROPAGATE_CUDA_ERROR(cudaMemcpy(scalers, h_scalers, sizeof(u32) * Number::LIMBS * len, cudaMemcpyHostToDevice));
        // Prepare for bucket sum
        Point *buckets_sum_buf;
        PROPAGATE_CUDA_ERROR(cudaMalloc(&buckets_sum_buf, sizeof(Point) * Config::n_windows * (Config::n_buckets - 1)));
        auto buckets_sum = Array2D<Point, Config::n_windows, Config::n_buckets - 1>(buckets_sum_buf);

        u32 *points;
        PROPAGATE_CUDA_ERROR(cudaMalloc(&points, sizeof(u32) * PointAffine::N_WORDS * len));
        PROPAGATE_CUDA_ERROR(cudaMemcpy(points, h_points, sizeof(u32) * PointAffine::N_WORDS * len, cudaMemcpyHostToDevice));   


        initalize_sum<Config, Point><<<Config::n_buckets - 1, Config::n_windows>>>(buckets_sum);
        PROPAGATE_CUDA_ERROR(cudaGetLastError());

        unsigned short int *mutex_buf;
        PROPAGATE_CUDA_ERROR(cudaMalloc(&mutex_buf, sizeof(unsigned short int) * Config::n_windows * (Config::n_buckets - 1)));
        PROPAGATE_CUDA_ERROR(cudaMemset(mutex_buf, 0, sizeof(unsigned short int) * Config::n_windows * (Config::n_buckets - 1)));
        Array2D<unsigned short int, Config::n_windows, Config::n_buckets - 1> mutex(mutex_buf);

        printf("hello\n");

        cudaEvent_t start,end;

        cudaEventCreate(&start);
        cudaEventCreate(&end);

        cudaEventRecord(start);

        direct_bucket_sum<Config><<<grid, Config::block_dim>>>(
            len,
            scalers,
            points,
            buckets_sum,
            mutex
        );

        PROPAGATE_CUDA_ERROR(cudaGetLastError());

        PROPAGATE_CUDA_ERROR(cudaEventRecord(end));
        PROPAGATE_CUDA_ERROR(cudaEventSynchronize(end));
        float ms;
        PROPAGATE_CUDA_ERROR(cudaEventElapsedTime(&ms, start, end));
        printf("sum: %f ms\n", ms);


        PROPAGATE_CUDA_ERROR(cudaFree(scalers));
        PROPAGATE_CUDA_ERROR(cudaFree(points));
        PROPAGATE_CUDA_ERROR(cudaFree(mutex_buf));

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
        usize temp_storage_bytes_reduce = 0;

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
