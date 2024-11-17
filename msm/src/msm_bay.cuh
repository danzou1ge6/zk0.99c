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
        static constexpr u32 s = 8;
        static constexpr u32 n_windows = div_ceil(lambda, s);
        static constexpr u32 n_buckets = pow2(s);
        static constexpr u32 n_window_id_bits = log2_ceil(n_windows);

        static constexpr u32 bay_size = 1 << 8;

        static constexpr bool debug = true;
    };

#ifdef DEBUG
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
#endif

    template <typename Config>
    struct Bay {
        u32 points[Config::bay_size];
    };
    struct __align__(16) BayMetaData{
        u32 window_id, scalar_id, num, bay_id;
        struct Decomposer{
            __host__ __device__ ::cuda::std::tuple<u32&, u32&> operator()(BayMetaData& meta_data) const {
                return {meta_data.window_id, meta_data.scalar_id};
            }
        };
    };

    // all bays are put in pool
    // bays, containing scalers (total windows * points / bay_size + blocks * windows * (buckets - 1) bays)
    template <typename Config, typename Field>
    __global__ void scatter_new(
        const u32* scalers,
        const u32 len,
        const u32 block_len,
        u32 *bay_allocator, // global atomic for allocating bays
        Bay<Config> *bay_buffer, // global buffer for bays
        BayMetaData *bay_meta_data, // global buffer for bay meta data
        u32 *bay_pointers, // local pointer for bays;
        Array2D<u32, Config::n_windows, Config::n_buckets - 1> bay_counts // count for bays in each bucket
    ) {
        const static usize WORDS = Field::LIMBS;
        using Number = mont::Number<WORDS>;
        
        Array2D<u32, Config::n_buckets - 1, Config::n_windows> bay_pointer(bay_pointers + (u64)blockIdx.x * (Config::n_buckets - 1) * Config::n_windows);

        __shared__ u32 bay_counters[Config::n_buckets - 1][Config::n_windows % 32 == 0 ? Config::n_windows + 1 : Config::n_windows];
        __shared__ u32 start_bay_id; // used to initialize the bays

        const u32 start_id = blockIdx.x * block_len + threadIdx.x;
        const u32 i_stride = blockDim.x;
        const u32 limit = min(block_len + blockIdx.x * block_len, len);

        // get inital bay buffer
        if (threadIdx.x == 0) {
            start_bay_id = atomicAdd(bay_allocator, Config::n_windows * (Config::n_buckets - 1));
        }
        __syncthreads();

        // Initialize local counters, and pointers
        if (Config::n_windows < Config::n_buckets - 1) {
            for (u32 bucket_id = threadIdx.x + 1; bucket_id < Config::n_buckets; bucket_id += blockDim.x) {
                #pragma unroll
                for (u32 window_id = 0; window_id < Config::n_windows; window_id++) {
                    auto bay_id = start_bay_id + (bucket_id - 1) * Config::n_windows + window_id;
                    auto meta_addr = bay_meta_data + bay_id;
                    bay_pointer.get(bucket_id - 1, window_id) = bay_id;
                    *meta_addr = BayMetaData{window_id, bucket_id, 0, bay_id};
                    bay_counters[bucket_id - 1][window_id] = 0;
                }
            }
        } else {
            for (u32 window_id = threadIdx.x; window_id < Config::n_windows; window_id += blockDim.x) {
                #pragma unroll
                for (u32 bucket_id = 1; bucket_id < Config::n_buckets; bucket_id++) {
                    auto bay_id = start_bay_id + (bucket_id - 1) * Config::n_windows + window_id;
                    auto meta_addr = bay_meta_data + bay_id;
                    bay_pointer.get(bucket_id - 1, window_id) = bay_id;
                    *meta_addr = BayMetaData{window_id, bucket_id, 0, bay_id};
                    bay_counters[bucket_id - 1][window_id] = 0;
                }
            }
        }
        
        __syncthreads();

        const u32 window_offset = threadIdx.x % Config::n_windows;
        // Scatter
        for (u32 i = start_id; i < limit; i += i_stride) {
            auto scaler = Number::load(scalers + i * WORDS);
            for (u32 j = 0; j < Config::n_windows; j++) {
                u32 window_id = (window_offset + j) % Config::n_windows; // rotate window id to avoid atomic conflict
                auto scaler_id = scaler.bit_slice(window_id * Config::s, Config::s);
                if (scaler_id != 0) {
                    scaler_id -= 1;
                    u32 index = atomicAdd_block(&bay_counters[scaler_id][window_id], 1);
                    __threadfence_block();
                    volatile u32 bay_id = bay_pointer.get(scaler_id, window_id);
                    
                    while (__any_sync(__activemask(), index >= Config::bay_size)){
                        if (index == Config::bay_size) {
                            (bay_meta_data + bay_id)->num = Config::bay_size;
                            atomicAdd(bay_counts.addr(window_id, scaler_id), 1);
                            // new bay
                            auto new_bay_id = atomicAdd(bay_allocator, 1);
                            auto meta_addr = bay_meta_data + new_bay_id;
                            *meta_addr = BayMetaData{window_id, scaler_id + 1, 0, new_bay_id};
                            bay_pointer.get(scaler_id, window_id) = new_bay_id;
                            __threadfence_block();
                            atomicExch_block(&bay_counters[scaler_id][window_id], 0);
                        }
                        if (index >= Config::bay_size) {
                            __nanosleep(1);
                            index = atomicAdd_block(&bay_counters[scaler_id][window_id], 1);
                            __threadfence_block();
                            bay_id = bay_pointer.get(scaler_id, window_id);
                        }
                    }
                    (bay_buffer + bay_id)->points[index] = i;
                }
            }
        }
        __syncthreads();

        // store the last bay
        if (Config::n_windows < Config::n_buckets - 1) {
            for (u32 bucket_id = threadIdx.x + 1; bucket_id < Config::n_buckets; bucket_id += blockDim.x) {
                #pragma unroll
                for (u32 window_id = 0; window_id < Config::n_windows; window_id++) {
                    auto meta_addr = bay_meta_data + bay_pointer.get(bucket_id - 1, window_id);
                    meta_addr->num = bay_counters[bucket_id - 1][window_id];
                    atomicAdd(bay_counts.addr(window_id, bucket_id - 1), 1);
                }
            }
        } else {
            for (u32 window_id = threadIdx.x; window_id < Config::n_windows; window_id += blockDim.x) {
                #pragma unroll
                for (u32 bucket_id = 1; bucket_id < Config::n_buckets; bucket_id++) {
                    auto meta_addr = bay_meta_data + bay_pointer.get(bucket_id - 1, window_id);
                    meta_addr->num = bay_counters[bucket_id - 1][window_id];
                    atomicAdd(bay_counts.addr(window_id, bucket_id - 1), 1);
                }
            }
        }
    }

   template <typename Config, typename Point, u32 WarpPerBlock>
    __global__ void bucket_sum_new(
        u32 max_task_step,
        u32 min_task_step,
        Bay<Config> *bay_buffer, // all bays
        BayMetaData *bay_meta_data, // meta data for each bay
        u32 *points, // all points
        u32 *bucket_allocator, // global atomic for allocating buckets
        u32 *finished_buckets, // number of finished buckets
        Array2D<u32, Config::n_windows, Config::n_buckets - 1> bay_allocators, // global atomic for allocating bays for each bucket
        u32 *bucket_offset, // offset for each bucket
        Array2D<Point, Config::n_windows, Config::n_buckets - 1> sum, // sum of each bucket
        Array2D<unsigned short int, Config::n_windows, Config::n_buckets - 1> mutex // mutex for each bucket
    ) {
        using WarpReduce = cub::WarpReduce<Point>;

        __shared__ typename WarpReduce::TempStorage temp_storage[WarpPerBlock];

        const u32 lane_id = threadIdx.x % THREADS_PER_WARP;
        const u32 warp_id = threadIdx.x / THREADS_PER_WARP;

        u32 index, window, bucket, cur_bay_cnt, task_step = max_task_step, meta_id, meta_end, offset;

        u32 can_return = 0;
        if (lane_id == 0) {
            do {
                if (*finished_buckets == Config::n_windows * (Config::n_buckets - 1)) {
                    can_return = 1;
                    break;
                }
                index = atomicInc(bucket_allocator, Config::n_windows * (Config::n_buckets - 1) - 1);
                window = index / (Config::n_buckets - 1);
                bucket = index % (Config::n_buckets - 1);
                cur_bay_cnt = bucket_offset[index + 1] - bucket_offset[index];
                task_step = max(task_step / 2, min_task_step);
                meta_id = atomicAdd(bay_allocators.addr(window, bucket), task_step);
            } while(meta_id >= cur_bay_cnt);
            if (meta_id < cur_bay_cnt && meta_id + task_step >= cur_bay_cnt) atomicAdd(finished_buckets, 1); // finish this bucket
        }

        // broadcast to all threads in the warp
        can_return = __shfl_sync(0xffffffff, can_return, 0);
        if (can_return == 1) return;
        index = __shfl_sync(0xffffffff, index, 0);
        window = index / (Config::n_buckets - 1);
        bucket = index % (Config::n_buckets - 1);
        cur_bay_cnt = __shfl_sync(0xffffffff, cur_bay_cnt, 0);
        task_step = __shfl_sync(0xffffffff, task_step, 0);
        meta_id = __shfl_sync(0xffffffff, meta_id, 0);
        meta_end = min(meta_id + task_step, cur_bay_cnt);
        offset = bucket_offset[index];

        BayMetaData meta_data;
        Point thread_sum = Point::identity();

        while(true) {
            meta_data = *(bay_meta_data + offset + meta_id);            
            u32 *bay_ptr = (bay_buffer + meta_data.bay_id)->points;

            for (u32 i = lane_id; i < meta_data.num; i += THREADS_PER_WARP) {
                auto point_id = bay_ptr[i];
                auto p = PointAffine::load(points + point_id * PointAffine::N_WORDS);
                thread_sum = thread_sum + p;
            }

            meta_id++;
            
            if (meta_id >= meta_end) {
                // the owned bays are all processed
                if (lane_id == 0) {
                    task_step = max(task_step / 2, min_task_step);
                    meta_id = atomicAdd(bay_allocators.addr(window, bucket), task_step);
                    if (meta_id < cur_bay_cnt && meta_id + task_step >= cur_bay_cnt) atomicAdd(finished_buckets, 1); // finish this bucket
                }
                meta_id = __shfl_sync(0xffffffff, meta_id, 0);
                if (meta_id < cur_bay_cnt) {
                    task_step = __shfl_sync(0xffffffff, task_step, 0);
                    meta_end = meta_end = min(meta_id + task_step, cur_bay_cnt);
                } else {
                    // switch to next bucket
                    // Sum up accumulated point from each thread in warp, returing the result to thread 0 in warp
                    Point bay_sum = WarpReduce(temp_storage[warp_id]).Reduce(thread_sum, [](const Point &a, const Point &b){ return a + b; });

                    if (lane_id == 0) {
                        auto mutex_ptr = mutex.addr(meta_data.window_id, meta_data.scalar_id);
                        u32 time = 1;
                        while (atomicCAS(mutex_ptr, (unsigned short int)0, (unsigned short int)1) != 0) {
                            __nanosleep(time);
                            if (time < 64) time *= 2;
                        }
                        __threadfence();
                        sum.get(window, bucket) = sum.get(window, bucket) + bay_sum;
                        __threadfence();
                        atomicCAS(mutex_ptr, (unsigned short int)1, (unsigned short int)0);

                        do {
                            if (*finished_buckets == Config::n_windows * (Config::n_buckets - 1)) {
                                can_return = 1;
                                break;
                            }
                            index = atomicInc(bucket_allocator, Config::n_windows * (Config::n_buckets - 1) - 1);
                            window = index / (Config::n_buckets - 1);
                            bucket = index % (Config::n_buckets - 1);
                            cur_bay_cnt = bucket_offset[index + 1] - bucket_offset[index];
                            task_step = max(task_step / 2, min_task_step);
                            meta_id = atomicAdd(bay_allocators.addr(window, bucket), task_step);
                        } while(meta_id >= cur_bay_cnt);
                        if (meta_id < cur_bay_cnt && meta_id + task_step >= cur_bay_cnt) atomicAdd(finished_buckets, 1); // finish this bucket
                    }
                    // broadcast to all threads in the warp
                    can_return = __shfl_sync(0xffffffff, can_return, 0);
                    if (can_return == 1) return;
                    index = __shfl_sync(0xffffffff, index, 0);
                    window = index / (Config::n_buckets - 1);
                    bucket = index % (Config::n_buckets - 1);
                    cur_bay_cnt = __shfl_sync(0xffffffff, cur_bay_cnt, 0);
                    task_step = __shfl_sync(0xffffffff, task_step, 0);
                    meta_id = __shfl_sync(0xffffffff, meta_id, 0);
                    meta_end = min(meta_id + task_step, cur_bay_cnt);
                    offset = bucket_offset[index];

                    thread_sum = Point::identity();
                }
            }
        }
    }


#ifdef DEBUG
    template <typename Config>
    __global__ void check_scatter(
        Bay<Config> *bay_buffer, 
        BayMetaData *bay_meta_data,
        u32 bay_num, 
        Array2D<u32, Config::n_windows, Config::n_buckets> counts,
        Array2D<u32, Config::n_windows, Config::n_buckets> counts_truth
    ) {
        for (u32 i = 0; i < bay_num; i++) {
            auto bay_ptr = bay_buffer + i;
            auto bay_meta_ptr = bay_meta_data + i;
            auto window_id = bay_meta_ptr->window_id;
            auto scalar_id = bay_meta_ptr->scalar_id;
            auto bay_id = bay_meta_ptr->bay_id;
            assert(bay_id == i);
            auto num = bay_meta_ptr->num;
            // printf("bay_id: %d, window_id: %d, scalar_id: %d, num: %d\n", bay_id, window_id, scalar_id, num);
            if (num > 0) {
                counts.get(window_id, scalar_id) += num;
            }
            auto ptr = bay_ptr->points;
            for (u32 j = 0; j < num; j++) {
                auto scaler = ptr[j];
                if (scaler == 0xffffffff) {
                    printf("Assertion failed! num %d window %d scalar %d i: %d, j: %d, value: %d\n",
                        num, window_id, scalar_id,i, j, scaler);
                    assert(false);
                }
            }
            for (u32 j = num; j < Config::bay_size; j++) {
                auto scaler = ptr[j];
                assert(scaler == 0xffffffff);
            }
        }
        for (int i = 0; i < Config::n_windows; i++) {
            for (int j = 0; j < Config::n_buckets; j++) {
                if (counts.get(i, j) != counts_truth.get(i, j)) {
                    printf("Assertion failed! i: %d, j: %d, value: %d, truth: %d\n",
                        i, j, counts.get(i, j), counts_truth.get(i, j));
                    assert(false);
                }
            }
        }
    }
#endif

    template <typename Config, typename Point>
    __global__ void initalize_sum(Array2D<Point, Config::n_windows, Config::n_buckets - 1> sum, u32 *bay_allocator) {
        u32 bucket_id = blockIdx.x;
        u32 window_id = threadIdx.x;

        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *bay_allocator = 0;
        }

        if (bucket_id >= Config::n_buckets || window_id >= Config::n_windows)
            return;

        sum.get(window_id, bucket_id) = Point::identity();
    }

    template <typename Config, typename Point>
    __global__ void naive_mul_points(Array2D<Point, Config::n_windows, Config::n_buckets - 1> sum) {
        u32 window_id = threadIdx.x;
        u32 bucket_id = blockIdx.x;

        if (bucket_id >= Config::n_buckets - 1 || window_id >= Config::n_windows)
            return;

        auto point = sum.get(window_id, bucket_id);

        point = point.multiple(bucket_id + 1);

        for (u32 i = 0; i < window_id * Config::s; i++) {
            point = point.self_add();
        }

        sum.get(window_id, bucket_id) = point;
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

    // A total of n_windows threads should be launched, only one block should be launched.
    // Opt. Opportunity: Perhaps this can be done faster by CPU; Or assign multiple threads to a window somehow
    template <typename Config>
    __global__ void bucket_reduction_and_window_reduction(
        const Array2D<Point, Config::n_windows, Config::n_buckets> buckets_sum,
        Point *reduced)
    {
        u32 window_id = threadIdx.x;

        auto acc = Point::identity();
        auto sum = Point::identity();

        // Bucket reduction
        // Perhaps save an iteration here by initializing acc to buckets_sum[n_buckets - 1][...]
        for (u32 i = Config::n_buckets - 1; i >= 1; i--)
        {
        acc = acc + buckets_sum.get_const(window_id, i);
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
    __host__ cudaError_t run(
        const u32 *h_scalers,
        const u32 *h_points,
        u32 len,
        Point &h_result)
    {
        cudaError_t err;

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        u32 grid = std::min(deviceProp.multiProcessorCount, (int)len);
        const u32 block = 512;

        // Count items in buckets
        u32 *scalers;
        PROPAGATE_CUDA_ERROR(cudaMalloc(&scalers, sizeof(u32) * Number::LIMBS * len));

        Bay<Config> *bay_buffer;
        BayMetaData *bay_meta_data;
        u64 max_bay_num = (div_ceil(Config::n_windows * len, Config::bay_size) + grid * Config::n_windows * (Config::n_buckets - 2));
        // printf("max_bay_num: %llu\n", max_bay_num);
        PROPAGATE_CUDA_ERROR(cudaMalloc(&bay_buffer, sizeof(Bay<Config>) * max_bay_num));
        PROPAGATE_CUDA_ERROR(cudaMalloc(&bay_meta_data, sizeof(BayMetaData) * max_bay_num));
#ifdef DEBUG
        if (Config::debug) {
            PROPAGATE_CUDA_ERROR(cudaMemset(bay_buffer, 0xff, sizeof(Bay<Config>) * max_bay_num));
        }
#endif
        PROPAGATE_CUDA_ERROR(cudaMemcpy(scalers, h_scalers, sizeof(u32) * Number::LIMBS * len, cudaMemcpyHostToDevice));

        u32 *allocator;
        PROPAGATE_CUDA_ERROR(cudaMalloc(&allocator, sizeof(u32)));
        PROPAGATE_CUDA_ERROR(cudaMemset(allocator, 0, sizeof(u32)));

        u32 *bay_pointers;
        PROPAGATE_CUDA_ERROR(cudaMalloc(&bay_pointers, sizeof(u32) * Config::n_windows * (Config::n_buckets - 1) * grid));

        u32 *bay_offset_buffer;
        PROPAGATE_CUDA_ERROR(cudaMalloc(&bay_offset_buffer, sizeof(u32) * (Config::n_windows * (Config::n_buckets - 1) + 1)));
        PROPAGATE_CUDA_ERROR(cudaMemset(bay_offset_buffer, 0, sizeof(u32) * (Config::n_windows * (Config::n_buckets - 1) + 1)));
        Array2D<u32, Config::n_windows, Config::n_buckets - 1> bay_counts(bay_offset_buffer);
        u32 block_len = div_ceil(len, grid);

        cudaEvent_t start,end;

        cudaEventCreate(&start);
        cudaEventCreate(&end);

        cudaEventRecord(start);

        scatter_new<Config, bn254_scalar::Element><<<grid, block>>>(
            scalers,
            len,
            block_len,
            allocator,
            bay_buffer,
            bay_meta_data,
            bay_pointers,
            bay_counts
        );

        PROPAGATE_CUDA_ERROR(cudaGetLastError());

        PROPAGATE_CUDA_ERROR(cudaEventRecord(end));
        PROPAGATE_CUDA_ERROR(cudaEventSynchronize(end));
        float ms;
        PROPAGATE_CUDA_ERROR(cudaEventElapsedTime(&ms, start, end));
        printf("scatter: %f ms\n", ms);

        int bay_num;
        PROPAGATE_CUDA_ERROR(cudaMemcpy(&bay_num, allocator, sizeof(u32), cudaMemcpyDeviceToHost));
        printf("bay_num: %d\n", bay_num);
#ifdef DEBUG
        if (Config::debug) {
            u32 *counts_buf, *counts_buf_truth;
            PROPAGATE_CUDA_ERROR(cudaMalloc(&counts_buf_truth, sizeof(u32) * Config::n_windows * Config::n_buckets));
            PROPAGATE_CUDA_ERROR(cudaMalloc(&counts_buf, sizeof(u32) * Config::n_windows * Config::n_buckets));
            Array2D<u32, Config::n_windows, Config::n_buckets> count(counts_buf), count_truth(counts_buf_truth);

            initialize_counts<Config><<<1, block>>>(count);
            initialize_counts<Config><<<1, block>>>(count_truth);
            count_buckets<Config><<<grid, block>>>(scalers, len, count_truth); 
            check_scatter<Config><<<1, 1>>>(bay_buffer, bay_meta_data, bay_num, count, count_truth);
            PROPAGATE_CUDA_ERROR(cudaGetLastError());
            PROPAGATE_CUDA_ERROR(cudaDeviceSynchronize());
            PROPAGATE_CUDA_ERROR(cudaFree(counts_buf));
            PROPAGATE_CUDA_ERROR(cudaFree(counts_buf_truth));
            
        }
#endif
        
        PROPAGATE_CUDA_ERROR(cudaFree(scalers));
        PROPAGATE_CUDA_ERROR(cudaFree(bay_pointers));

        // perfix sum for bay_num
        void *d_temp_storage_scan = nullptr;
        usize temp_storage_bytes_scan = 0;

        PROPAGATE_CUDA_ERROR(
            cub::DeviceScan::ExclusiveSum(
                d_temp_storage_scan, 
                temp_storage_bytes_scan, 
                bay_offset_buffer, 
                bay_offset_buffer, 
                Config::n_windows * (Config::n_buckets - 1) + 1
            )
        )

        PROPAGATE_CUDA_ERROR(cudaMalloc(&d_temp_storage_scan, temp_storage_bytes_scan));
        PROPAGATE_CUDA_ERROR(cudaEventRecord(start));

        PROPAGATE_CUDA_ERROR(
            cub::DeviceScan::ExclusiveSum(
                d_temp_storage_scan, 
                temp_storage_bytes_scan, 
                bay_offset_buffer, 
                bay_offset_buffer, 
                Config::n_windows * (Config::n_buckets - 1) + 1
            )
        )
        PROPAGATE_CUDA_ERROR(cudaFree(d_temp_storage_scan));

        PROPAGATE_CUDA_ERROR(cudaEventRecord(end));
        PROPAGATE_CUDA_ERROR(cudaEventSynchronize(end));
        PROPAGATE_CUDA_ERROR(cudaEventElapsedTime(&ms, start, end));
        printf("prefix sum: %f ms\n", ms);

        // Sort all the bays
        void *d_temp_storage = nullptr;
        usize temp_storage_bytes = 0;

        PROPAGATE_CUDA_ERROR(
            cub::DeviceRadixSort::SortKeys(
                d_temp_storage, 
                temp_storage_bytes, 
                bay_meta_data, 
                bay_meta_data, 
                bay_num, 
                BayMetaData::Decomposer()
            )
        );
        PROPAGATE_CUDA_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));

        PROPAGATE_CUDA_ERROR(cudaEventRecord(start));

        PROPAGATE_CUDA_ERROR(
            cub::DeviceRadixSort::SortKeys(
                d_temp_storage, 
                temp_storage_bytes, 
                bay_meta_data, 
                bay_meta_data, 
                bay_num, 
                BayMetaData::Decomposer()
            )
        );

        PROPAGATE_CUDA_ERROR(cudaFree(d_temp_storage));

        PROPAGATE_CUDA_ERROR(cudaEventRecord(end));
        PROPAGATE_CUDA_ERROR(cudaEventSynchronize(end));
        PROPAGATE_CUDA_ERROR(cudaEventElapsedTime(&ms, start, end));
        printf("sort: %f ms\n", ms);

#ifdef DEBUG
        // copy bay_meta_data to host and print when debug
        // if (Config::debug) {
        //     BayMetaData *h_bay_meta_data = (BayMetaData *)malloc(sizeof(BayMetaData) * bay_num);
        //     PROPAGATE_CUDA_ERROR(cudaMemcpy(h_bay_meta_data, bay_meta_data, sizeof(BayMetaData) * bay_num, cudaMemcpyDeviceToHost));
        
        //     for (int i = 0; i < bay_num; i++) {
        //         printf("bay_id: %d, window_id: %d, scalar_id: %d, num: %d\n", h_bay_meta_data[i].bay_id, h_bay_meta_data[i].window_id, h_bay_meta_data[i].scalar_id, h_bay_meta_data[i].num);
        //     }
        // }
#endif

        // Prepare for bucket sum
        Point *buckets_sum_buf;
        PROPAGATE_CUDA_ERROR(cudaMalloc(&buckets_sum_buf, sizeof(Point) * Config::n_windows * (Config::n_buckets - 1)));
        auto buckets_sum = Array2D<Point, Config::n_windows, Config::n_buckets - 1>(buckets_sum_buf);

        u32 *points;
        PROPAGATE_CUDA_ERROR(cudaMalloc(&points, sizeof(u32) * PointAffine::N_WORDS * len));
        PROPAGATE_CUDA_ERROR(cudaMemcpy(points, h_points, sizeof(u32) * PointAffine::N_WORDS * len, cudaMemcpyHostToDevice));   


        initalize_sum<Config, Point><<<Config::n_buckets - 1, Config::n_windows>>>(buckets_sum, allocator);
        PROPAGATE_CUDA_ERROR(cudaGetLastError());

        unsigned short int *mutex_buf;
        PROPAGATE_CUDA_ERROR(cudaMalloc(&mutex_buf, sizeof(unsigned short int) * Config::n_windows * (Config::n_buckets - 1)));
        PROPAGATE_CUDA_ERROR(cudaMemset(mutex_buf, 0, sizeof(unsigned short int) * Config::n_windows * (Config::n_buckets - 1)));

        u32 *finished_buckets;
        PROPAGATE_CUDA_ERROR(cudaMalloc(&finished_buckets, sizeof(u32)));
        PROPAGATE_CUDA_ERROR(cudaMemset(finished_buckets, 0, sizeof(u32)));

        u32 *bay_allocator_buffer;
        PROPAGATE_CUDA_ERROR(cudaMalloc(&bay_allocator_buffer, sizeof(u32) * Config::n_windows * (Config::n_buckets - 1)));
        PROPAGATE_CUDA_ERROR(cudaMemset(bay_allocator_buffer, 0, sizeof(u32) * Config::n_windows * (Config::n_buckets - 1)));
        Array2D<u32, Config::n_windows, Config::n_buckets - 1> bay_allocators(bay_allocator_buffer);

        Array2D<unsigned short int, Config::n_windows, Config::n_buckets - 1> mutex(mutex_buf);
        
        u32 max_task_step = bay_num / (grid * 256 / THREADS_PER_WARP) / 2;
        u32 min_task_step = div_ceil(Config::bay_size, div_ceil(len * Config::n_windows, bay_num));
        printf("max_task_step: %d, min_task_step: %d\n", max_task_step, min_task_step);
        cudaEventRecord(start);
        bucket_sum_new<Config, Point, 256 / THREADS_PER_WARP><<<grid, 256>>>(
            max_task_step,
            min_task_step,
            bay_buffer,
            bay_meta_data, 
            points, 
            allocator, 
            finished_buckets, 
            bay_allocators, 
            bay_offset_buffer,
            buckets_sum, 
            mutex
        );
        PROPAGATE_CUDA_ERROR(cudaGetLastError());
        cudaEventRecord(end);
        PROPAGATE_CUDA_ERROR(cudaEventSynchronize(end));
        cudaEventElapsedTime(&ms, start, end);
        printf("bucket sum: %f ms\n", ms);

        PROPAGATE_CUDA_ERROR(cudaFree(points));
        PROPAGATE_CUDA_ERROR(cudaFree(bay_buffer));
        PROPAGATE_CUDA_ERROR(cudaFree(bay_meta_data));
        PROPAGATE_CUDA_ERROR(cudaFree(allocator));
        PROPAGATE_CUDA_ERROR(cudaFree(mutex_buf));
        PROPAGATE_CUDA_ERROR(cudaFree(finished_buckets));
        PROPAGATE_CUDA_ERROR(cudaFree(bay_allocator_buffer));

        PROPAGATE_CUDA_ERROR(cudaEventRecord(start));
        
        naive_mul_points<Config><<<Config::n_buckets - 1, Config::n_windows>>>(buckets_sum);

        PROPAGATE_CUDA_ERROR(cudaEventRecord(end));
        PROPAGATE_CUDA_ERROR(cudaEventSynchronize(end));
        PROPAGATE_CUDA_ERROR(cudaEventElapsedTime(&ms, start, end));
        printf("Point multiply: %f ms\n", ms);

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
                buckets_sum_buf, 
                reduced, 
                Config::n_windows * (Config::n_buckets - 1), 
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
                buckets_sum_buf, 
                reduced, 
                Config::n_windows * (Config::n_buckets - 1), 
                add_op,
                Point::identity()
            )
        );
        PROPAGATE_CUDA_ERROR(cudaEventRecord(end));
        PROPAGATE_CUDA_ERROR(cudaEventSynchronize(end));
        PROPAGATE_CUDA_ERROR(cudaEventElapsedTime(&ms, start, end));
        printf("reduce: %f ms\n", ms);

        PROPAGATE_CUDA_ERROR(cudaFree(d_temp_storage_reduce));

        PROPAGATE_CUDA_ERROR(cudaMemcpy(&h_result, reduced, sizeof(Point), cudaMemcpyDeviceToHost));
        PROPAGATE_CUDA_ERROR(cudaFree(reduced));

        return cudaSuccess;
    }

}
