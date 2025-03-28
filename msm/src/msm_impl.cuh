#include "msm.cuh"
#include "../../mont/src/bn254_scalar.cuh"
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <array>
#include <chrono>
#include <cub/cub.cuh>
#include <iostream>
#include <thread>

#define PROPAGATE_CUDA_ERROR(x)                                                                                    \
{                                                                                                                \
    err = x;                                                                                                       \
    if (err != cudaSuccess)                                                                                        \
    {                                                                                                              \
        std::cerr << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: " << cudaGetErrorString(err) << " (Error Code: " << err << ")" << std::endl; \
        return err;                                                                                                  \
    }                                                                                                              \
}

#ifdef __CUDA_ARCH__
#define likely(x) (__builtin_expect((x), 1))
#define unlikely(x) (__builtin_expect((x), 0))
#else
#define likely(x) (x) [[likely]]
#define unlikely(x) (x) [[unlikely]]
#endif 

#define TPI 2

namespace msm {

    using Params1 = bn254_scalar::Params;

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
    template <typename Config, typename Element>
    __global__ void distribute_windows(
        const u32 *scalers,
        const u64 len,
        u32* cnt_zero,
        u64* indexs,
        u32* points_offset
    ) {
        u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
        u32 stride = gridDim.x * blockDim.x;

        u32 cnt_zero_local = 0;
        
        // Count into block-wide counter
        for (u32 i = tid; i < len; i += stride) {
            int bucket[Config::actual_windows];
            auto scaler = Element::load(scalers + i * Element::LIMBS);
            scaler.bit_slice<Config::actual_windows, Config::s>(bucket);
            signed_digit<Config::actual_windows, Config::s>(bucket);

            #pragma unroll
            for (u32 window_id = 0; window_id < Config::actual_windows; window_id++) {
                auto sign = bucket[window_id] < 0;
                auto bucket_id = sign ? -bucket[window_id] : bucket[window_id];
                // if(tid == 0 && i < 5) {
                //     printf("%d %d: %d\n", i, window_id, bucket_id);
                // }
                u32 physical_window_id = window_id % Config::n_windows;
                u32 point_group = window_id / Config::n_windows;
                if (bucket_id == 0) {
                    cnt_zero_local++;
                }
                u64 index = bucket_id | (sign << Config::s) | (physical_window_id << (Config::s + 1)) 
                | ((point_group * len + i) << (Config::s + 1 + Config::window_bits));
                indexs[(points_offset[physical_window_id] + point_group) * len + i] = index;
            }
        }
        if(cnt_zero_local > 0)
            atomicAdd(cnt_zero, cnt_zero_local);
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

    template <typename Config, u32 WarpPerBlock, typename Point, typename PointAffine>
    __global__ void bucket_sum(
        const u64 len,
        const u32 *cnt_zero,
        const u64 *indexs,
        const u32 *points,
        Array2D<unsigned short, Config::n_windows, Config::n_buckets> mutex,
        Array2D<unsigned short, Config::n_windows, Config::n_buckets> initialized,
        Array2D<Point, Config::n_windows, Config::n_buckets * TPI> sum
    ) {
        __shared__ u32 point_buffer[WarpPerBlock * THREADS_PER_WARP][PointAffine::N_WORDS * 2 + 4]; // +4 for padding and alignment
        const static u32 key_mask = (1u << Config::s) - 1;
        const static u32 sign_mask = 1u << Config::s;
        const static u32 window_mask = (1u << Config::window_bits) - 1;

        const u32 gtid = (threadIdx.x + blockIdx.x * blockDim.x) / TPI;
        const u32 threads = (blockDim.x * gridDim.x) / TPI;
        const u32 group_thread = threadIdx.x & (TPI-1);
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
        
        // used to special handle the last bucket
        u64 last_index = indexs[end_id - 1];
        u32 last_key = last_index & key_mask;
        u32 last_window_id = (last_index >> (Config::s + 1)) & window_mask;

        pip_thread.producer_acquire();
        cuda::memcpy_async(smem_ptr0, reinterpret_cast<const uint4*>(points + pointer * PointAffine::N_WORDS * TPI + group_thread * PointAffine::N_WORDS), sizeof(PointAffine), pip_thread);
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
            cuda::memcpy_async(g2s_ptr, reinterpret_cast<const uint4*>(points + pointer * PointAffine::N_WORDS * TPI + group_thread * PointAffine::N_WORDS), sizeof(PointAffine), pip_thread);
            pip_thread.producer_commit();
            stage ^= 1;
            
            cuda::pipeline_consumer_wait_prior<1>(pip_thread);
            PointAffine p = PointAffine::load(reinterpret_cast<u32*>(s2r_ptr));
            pip_thread.consumer_release();

            if (sign) p = p.neg();
            acc = acc + p;
            // acc = acc.add_pre(p);

            u32 next_key = next_index & key_mask;
            u32 next_window_id = (next_index >> (Config::s + 1)) & window_mask;

            if unlikely(next_key != key || next_window_id != window_id) {
                if unlikely(first) {                    
                    unsigned short *mutex_ptr;
                    mutex_ptr = mutex.addr(window_id, key - 1);
                    if(group_thread == 0)
                        lock(mutex_ptr);
                    if (initialized.get(window_id, key - 1)) {
                        sum.get(window_id, (key - 1) * TPI + group_thread) = sum.get(window_id, (key - 1) * TPI + group_thread) + acc;
                        // sum.get(window_id, key - 1) = sum.get(window_id, key - 1).add_pre(acc);
                    } else {
                        sum.get(window_id, (key - 1) * TPI + group_thread) = acc;
                        if(group_thread == 0)
                            initialized.get(window_id, key - 1) = 1;
                    }
                    if(group_thread == 0)
                        unlock(mutex_ptr);
                    first = false;
                }
                else {
                    sum.get(window_id, (key - 1) * TPI + group_thread) = acc;
                    if(group_thread == 0)
                        initialized.get(window_id, key - 1) = 1;
                }

                // __syncwarp();

                if (initialized.get(next_window_id, next_key - 1) && (next_key != last_key || next_window_id != last_window_id)) {
                    acc = sum.get(next_window_id, (next_key - 1) * TPI + group_thread);
                } else {
                    acc = Point::identity();
                }
            }
            key = next_key;
            sign = (next_index & sign_mask) != 0;
            window_id = next_window_id;
        }

        pip_thread.consumer_wait();
        PointAffine p = PointAffine::load(reinterpret_cast<u32*>(stage == 0 ? smem_ptr1 : smem_ptr0));
        pip_thread.consumer_release();

        if (sign) p = p.neg();
        acc = acc + p;
        // acc = acc.add_pre(p);
        
        auto mutex_ptr = mutex.addr(window_id, key - 1);
        if(group_thread == 0)
            lock(mutex_ptr);
        if (initialized.get(window_id, key - 1)) {
            sum.get(window_id, (key - 1) * TPI + group_thread) = sum.get(window_id, (key - 1) * TPI + group_thread) + acc;
            // sum.get(window_id, key - 1) = sum.get(window_id, key - 1).add_pre(acc);
        } else {
            sum.get(window_id, (key - 1) * TPI + group_thread) = acc;
            if(group_thread == 0)
                initialized.get(window_id, key - 1) = 1;
        }
        if(group_thread == 0)
            unlock(mutex_ptr);
        // __syncwarp();
    }

    template<typename Config, u32 WarpPerBlock, typename Point>
    __launch_bounds__(256,1)
    __global__ void reduceBuckets(
        Array2D<Point, Config::n_windows, Config::n_buckets * TPI> buckets_sum, 
        Point *reduceMemory,
        Array2D<unsigned short, Config::n_windows, Config::n_buckets> initialized
    ) {

        assert(gridDim.x % Config::n_windows == 0);

        __shared__ u32 smem[WarpPerBlock * TPI][Point::N_WORDS + 4]; // +4 for padding

        const u32 total_threads_per_window = gridDim.x / Config::n_windows * blockDim.x / TPI;
        u32 window_id = blockIdx.x / (gridDim.x / Config::n_windows);

        u32 wtid = ((blockIdx.x % (gridDim.x / Config::n_windows)) * blockDim.x + threadIdx.x) / TPI;
          
        const u32 buckets_per_thread = div_ceil(Config::n_buckets, total_threads_per_window);
        u32 group_thread = threadIdx.x & (TPI-1);

        Point sum, sum_of_sums;

        sum = Point::identity();
        sum_of_sums = Point::identity();

        // if(blockIdx.x == 0 && threadIdx.x == 0) {
        //     for(int i = 0; i < 2; ++i) {
        //         for(int j=1; j < 3; ++j) {
        //             printf("%d %d\n", i, j);
        //             buckets_sum.get(i, j - 1).device_print();
        //         }
        //     }
        // }

        for(u32 i=buckets_per_thread; i > 0; i--) {
            u32 loadIndex = wtid * buckets_per_thread + i;
            if(loadIndex <= Config::n_buckets && initialized.get(window_id, loadIndex - 1)) {
                sum = sum + buckets_sum.get(window_id, (loadIndex - 1) * TPI + group_thread);
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

        for(int i=16; i>=TPI; i=i/2) {
            sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(i);
        }

        if(lane_id < TPI) {
            sum_of_sums.store(smem[warp_id * TPI + group_thread]);
        }

        __syncthreads();

        if (warp_id > 0) return;

        if (threadIdx.x < WarpPerBlock * TPI) {
            sum_of_sums = Point::load(smem[threadIdx.x]);
        } else {
            sum_of_sums = Point::identity();
        }

        // Reduce in warp1
        if constexpr (TPI <= 16 && WarpPerBlock * TPI > 16) sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(16);
        if constexpr (TPI <= 8 && WarpPerBlock * TPI > 8) sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(8);
        if constexpr (TPI <= 4 && WarpPerBlock * TPI > 4) sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(4);
        if constexpr (TPI <= 2 && WarpPerBlock * TPI > 2) sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(2);
        if constexpr (TPI <= 1 && WarpPerBlock * TPI > 1) sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(1);

        // Store to global memory    
        if (threadIdx.x < TPI) {
            for (u32 i = 0; i < window_id * Config::s; i++) {
                sum_of_sums = sum_of_sums.self_add();
            }
            reduceMemory[blockIdx.x * TPI + group_thread] = sum_of_sums;
        }        
    }

    template <typename Config, typename Point, typename PointAffine>
    __global__ void precompute_kernel(u32 *points, u64 len) {
        // u64 gid = threadIdx.x + blockIdx.x * blockDim.x;
        // if (gid >= len) return;
        // auto p = PointAffine::load(points + gid * PointAffine::N_WORDS).to_point();
        // for (u32 i = 1; i < Config::n_precompute; i++) {
        //     #pragma unroll
        //     for (u32 j = 0; j < Config::n_windows * Config::s; j++) {
        //         p = p.self_add();
        //         // p = p.self_add_pre();
        //     }
        //     p.to_affine().store(points + (gid + i * len * TPI) * PointAffine::N_WORDS);
        // }
    }

    template <typename Config, typename Element, typename Point, typename PointAffine, typename PointAll>
    cudaError_t MSM<Config, Element, Point, PointAffine, PointAll>::run(const u32 batches, std::vector<u32*>::const_iterator h_scalers, bool first_run, cudaStream_t stream) {
        cudaError_t err;

        u64 part_len = div_ceil(len, parts);

        for (u32 i = 0; i < batches; i++) {
            PROPAGATE_CUDA_ERROR(cudaMemsetAsync(initialized_buf[i], 0, sizeof(unsigned short) * (Config::n_buckets) * Config::n_windows, stream));
        }

        auto mutex = Array2D<unsigned short, Config::n_windows, Config::n_buckets>(mutex_buf);

        int begin = 0, end = parts, stride = 1;

        if (head) {
            begin = 0;
            end = parts;
            stride = 1;
            head = false;
        } else {
            begin = parts - 1;
            end = -1;
            stride = -1;
            head = true;
        }
        u32 first_len = std::min(part_len, len - begin * part_len);
        u32 points_transported = first_run ? 0 : first_len;
        
        cudaStream_t copy_stream;
        PROPAGATE_CUDA_ERROR(cudaStreamCreate(&copy_stream));
        for (u32 i = 0; i < max_scaler_stages; i++) {
            PROPAGATE_CUDA_ERROR(cudaEventRecord(begin_scaler_copy[i], stream));
        }
        for (u32 i = 0; i < max_point_stages; i++) {
            PROPAGATE_CUDA_ERROR(cudaEventRecord(begin_point_copy[i], stream));
        }
        
        u32 points_per_transfer;

        for (int p = begin; p != end; p += stride) {
            u64 offset = p * part_len;
            u64 cur_len = std::min(part_len, len - offset);
            if (p + stride != end) {
                auto next_len = std::min(part_len, len - (p + stride) * part_len);
                points_per_transfer = div_ceil(next_len, batches);
            }

            for (int j = 0; j < batches; j++) {
                cudaEvent_t start1, stop1;
                float elapsedTime1 = 0.0;
                cudaEventCreate(&start1);
                cudaEventCreate(&stop1);
                PROPAGATE_CUDA_ERROR(cudaStreamWaitEvent(copy_stream, begin_scaler_copy[stage_scaler], cudaEventWaitDefault));
                if constexpr (Config::debug) {
                    cudaEventRecord(start1, copy_stream);
                }
                PROPAGATE_CUDA_ERROR(cudaMemcpyAsync(scalers[stage_scaler], *(h_scalers + j) + offset * Config::LIMBS, sizeof(u32) * Config::LIMBS * cur_len, cudaMemcpyHostToDevice, copy_stream));
                if constexpr (Config::debug) {
                    cudaEventRecord(stop1, copy_stream);
                    PROPAGATE_CUDA_ERROR(cudaEventSynchronize(stop1));
                    cudaEventElapsedTime(&elapsedTime1, start1, stop1);
                    std::cout << "Scaler transfering time:" << elapsedTime1 << std::endl;
                }
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
                // printf("Stage3\n");
                u32 block_size = 512;
                u32 grid_size = num_sm;
                distribute_windows<Config, Element><<<grid_size, block_size, 0, stream>>>(
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
                // printf("Stage4\n");
                cub::DeviceRadixSort::SortKeys(
                    d_temp_storage_sort, temp_storage_bytes_sort,
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
                // printf("Stage5\n");
                // wait before the first point copy
                if (points_transported == 0) PROPAGATE_CUDA_ERROR(cudaStreamWaitEvent(copy_stream, begin_point_copy[stage_point_transporting], cudaEventWaitDefault));
                if constexpr (Config::debug) {
                    cudaEventRecord(start1, copy_stream);
                }
                if (j == 0) {
                    u32 point_left = cur_len - points_transported;
                    if (point_left > 0) for (int i = 0; i < Config::n_precompute; i++) {
                        PROPAGATE_CUDA_ERROR(cudaMemcpyAsync(
                            points[stage_point_transporting] + (i * cur_len + points_transported) * PointAffine::N_WORDS * TPI,
                            h_points[i] + (offset + points_transported) * PointAffine::N_WORDS * TPI,
                            sizeof(PointAffine) * point_left * TPI, cudaMemcpyHostToDevice, copy_stream
                        ));
                    }
                    PROPAGATE_CUDA_ERROR(cudaEventRecord(end_point_copy[stage_point_transporting], copy_stream));
                    points_transported = 0;
                    if (p + stride != end) stage_point_transporting = (stage_point_transporting + 1) % max_point_stages;
                } else if(p + stride != end) {
                    u64 next_offset = (p + stride) * part_len;

                    for (int i = 0; i < Config::n_precompute; i++) {
                        PROPAGATE_CUDA_ERROR(cudaMemcpyAsync(
                            points[stage_point_transporting] + (i * cur_len + points_transported) * PointAffine::N_WORDS * TPI,
                            h_points[i] + (next_offset + points_transported) * PointAffine::N_WORDS * TPI,
                            sizeof(PointAffine) * points_per_transfer * TPI, cudaMemcpyHostToDevice, copy_stream
                        ));
                    }
                    points_transported += points_per_transfer;
                }
                if constexpr (Config::debug) {
                    cudaEventRecord(stop1, copy_stream);
                    PROPAGATE_CUDA_ERROR(cudaEventSynchronize(stop1));
                    cudaEventElapsedTime(&elapsedTime1, start1, stop1);
                    std::cout << "Point transfering time:" << elapsedTime1 << std::endl;
                }

                if (j == 0) PROPAGATE_CUDA_ERROR(cudaStreamWaitEvent(stream, end_point_copy[stage_point], cudaEventWaitDefault));

                if constexpr (Config::debug) {
                    cudaEventRecord(start, stream);
                }

                // Do bucket sum
                block_size = 256;
                grid_size = num_sm;
                // printf("Stage6\n");
                bucket_sum<Config, 8, Point, PointAffine><<<grid_size, block_size, 0, stream>>>(
                    cur_len * Config::actual_windows,
                    cnt_zero,
                    indexs,
                    points[stage_point],
                    mutex,
                    initialized[j],
                    buckets_sum[j]
                );
                PROPAGATE_CUDA_ERROR(cudaGetLastError());

                if constexpr (Config::debug) {
                    cudaEventRecord(stop, stream);
                    PROPAGATE_CUDA_ERROR(cudaEventSynchronize(stop));
                    cudaEventElapsedTime(&elapsedTime, start, stop);
                    std::cout << "MSM bucket sum time:" << elapsedTime << std::endl;
                }
                // printf("Stage7\n");

                if (j == batches - 1) PROPAGATE_CUDA_ERROR(cudaEventRecord(begin_point_copy[stage_point], stream));

                stage_scaler = (stage_scaler + 1) % max_scaler_stages;
            }

            if (p + stride != end) {
                stage_point = (stage_point + 1) % max_point_stages;
            }
        }

        PROPAGATE_CUDA_ERROR(cudaStreamDestroy(copy_stream));

        cudaEvent_t start, stop;
        float ms;

        if constexpr (Config::debug) {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
        }

        u32 grid = reduce_blocks; 
        // printf("Stage8\n");
        // start reduce
        for (int j = 0; j < batches; j++) {
            if constexpr (Config::debug) {
                cudaEventRecord(start, stream);
            }
            
            reduceBuckets<Config, 8, Point> <<< grid, 256, 0, stream >>> (buckets_sum[j], reduce_buffer[j], initialized[j]);

            PROPAGATE_CUDA_ERROR(cudaGetLastError());

            if constexpr (Config::debug) {
                cudaEventRecord(stop, stream);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&ms, start, stop);
                std::cout << "MSM bucket reduce time:" << ms << std::endl;
            }
        }
        // printf("Stage9\n");
        return cudaSuccess;
        }
 
    template <typename Config, typename Element, typename Point, typename PointAffine, typename PointAll>
    MSM<Config, Element, Point, PointAffine, PointAll>::MSM(u64 len, u32 batch_per_run, u32 parts, u32 scaler_stages, u32 point_stages, int device)
    : len(len), batch_per_run(batch_per_run), parts(parts), max_scaler_stages(scaler_stages), max_point_stages(point_stages),
    stage_scaler(0), stage_point(0), stage_point_transporting(0), device(device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        num_sm = deviceProp.multiProcessorCount;
        reduce_blocks = div_ceil(num_sm, Config::n_windows) * Config::n_windows;
        d_buckets_sum_buf = new Point*[batch_per_run];
        buckets_sum = new Array2D<Point, Config::n_windows, Config::n_buckets * TPI>[batch_per_run];
        initialized_buf = new unsigned short*[batch_per_run];
        initialized = new Array2D<unsigned short, Config::n_windows, Config::n_buckets>[batch_per_run];
        scalers = new u32*[scaler_stages];
        points = new u32*[point_stages];
        begin_scaler_copy = new cudaEvent_t[scaler_stages];
        end_scaler_copy = new cudaEvent_t[scaler_stages];
        begin_point_copy = new cudaEvent_t[point_stages];
        end_point_copy = new cudaEvent_t[point_stages];
        reduce_buffer = new Point*[batch_per_run];
        h_reduce_buffer = new Point*[batch_per_run];
        
        for (int j = 0; j < batch_per_run; j++) {
            cudaMallocHost(&h_reduce_buffer[j], sizeof(Point) * reduce_blocks * TPI, cudaHostAllocDefault);
        }
        h_points_offset[0] = 0;
        for (u32 i = 0; i < Config::n_windows; i++) {
            h_points_per_window[i] = div_ceil(Config::actual_windows - i, Config::n_windows);
            h_points_offset[i + 1] = h_points_offset[i] + h_points_per_window[i];
        }
        // points_offset[Config::n_windows] should be the total number of points
        assert(h_points_offset[Config::n_windows] == Config::actual_windows);
    }

    template <typename Config, typename Element, typename Point, typename PointAffine, typename PointAll>
    void MSM<Config, Element, Point, PointAffine, PointAll>::set_points(std::array<u32*, Config::n_precompute> host_points) {
        for (u32 i = 0; i < Config::n_precompute; i++) {
            h_points[i] = host_points[i];
        }
        points_set = true;
    }

    template <typename Config, typename Element, typename Point, typename PointAffine, typename PointAll>
    cudaError_t MSM<Config, Element, Point, PointAffine, PointAll>::alloc_gpu(cudaStream_t stream) {
        cudaError_t err;
        PROPAGATE_CUDA_ERROR(cudaSetDevice(device));
        u64 part_len = div_ceil(len, parts);
        for (u32 i = 0; i < batch_per_run; i++) {
            PROPAGATE_CUDA_ERROR(cudaMallocAsync(&d_buckets_sum_buf[i], sizeof(Point) * Config::n_windows * Config::n_buckets * TPI, stream));
            buckets_sum[i] = Array2D<Point, Config::n_windows, Config::n_buckets * TPI>(d_buckets_sum_buf[i]);
            PROPAGATE_CUDA_ERROR(cudaMallocAsync(&initialized_buf[i], sizeof(unsigned short) * (Config::n_buckets) * Config::n_windows, stream));
            initialized[i] = Array2D<unsigned short, Config::n_windows, Config::n_buckets>(initialized_buf[i]);
        }
        PROPAGATE_CUDA_ERROR(cudaMallocAsync(&mutex_buf, sizeof(unsigned short) * (Config::n_buckets) * Config::n_windows, stream));
        PROPAGATE_CUDA_ERROR(cudaMemsetAsync(mutex_buf, 0, sizeof(unsigned short) * (Config::n_buckets) * Config::n_windows, stream));
        PROPAGATE_CUDA_ERROR(cudaMallocAsync(&d_points_offset, sizeof(u32) * (Config::n_windows + 1), stream));
        PROPAGATE_CUDA_ERROR(cudaMemcpyAsync(d_points_offset, h_points_offset, sizeof(u32) * (Config::n_windows + 1), cudaMemcpyHostToDevice, stream));
        PROPAGATE_CUDA_ERROR(cudaMallocAsync(&cnt_zero, sizeof(u32), stream));
        PROPAGATE_CUDA_ERROR(cudaMallocAsync(&indexs, sizeof(u64) * Config::actual_windows * part_len * 2, stream));
        for (int i = 0; i < max_scaler_stages; i++) {
            PROPAGATE_CUDA_ERROR(cudaMallocAsync(&scalers[i], sizeof(u32) * Config::LIMBS * part_len, stream));
        }
        for (int i = 0; i < max_point_stages; i++) {
            PROPAGATE_CUDA_ERROR(cudaMallocAsync(&points[i], sizeof(u32) * PointAffine::N_WORDS * part_len * Config::n_precompute * TPI, stream));
        }
        cub::DeviceRadixSort::SortKeys(
            nullptr, temp_storage_bytes_sort,
            indexs + part_len * Config::actual_windows, indexs,
            Config::actual_windows * part_len, 0, Config::s, stream
        );
        PROPAGATE_CUDA_ERROR(cudaMallocAsync(&d_temp_storage_sort, temp_storage_bytes_sort, stream));
        for (int i = 0; i < batch_per_run; i++) {
            PROPAGATE_CUDA_ERROR(cudaMallocAsync(&reduce_buffer[i], sizeof(Point) * reduce_blocks * TPI, stream));
        }
        for (int i = 0; i < max_scaler_stages; i++) {
            PROPAGATE_CUDA_ERROR(cudaEventCreate(begin_scaler_copy + i));
            PROPAGATE_CUDA_ERROR(cudaEventCreate(end_scaler_copy + i));
        }
        for (int i = 0; i < max_point_stages; i++) {
            PROPAGATE_CUDA_ERROR(cudaEventCreate(begin_point_copy + i));
            PROPAGATE_CUDA_ERROR(cudaEventCreate(end_point_copy + i));
        }
        allocated = true;
        return cudaSuccess;
    }

    template <typename Config, typename Element, typename Point, typename PointAffine, typename PointAll>
    cudaError_t MSM<Config, Element, Point, PointAffine, PointAll>::free_gpu(cudaStream_t stream) {
        cudaError_t err;
        PROPAGATE_CUDA_ERROR(cudaSetDevice(device));
        if (!allocated) return cudaSuccess;
        for (u32 i = 0; i < batch_per_run; i++) {
            PROPAGATE_CUDA_ERROR(cudaFreeAsync(d_buckets_sum_buf[i], stream));
            PROPAGATE_CUDA_ERROR(cudaFreeAsync(initialized_buf[i], stream));
        }
        PROPAGATE_CUDA_ERROR(cudaFreeAsync(mutex_buf, stream));
        PROPAGATE_CUDA_ERROR(cudaFreeAsync(d_points_offset, stream));
        PROPAGATE_CUDA_ERROR(cudaFreeAsync(cnt_zero, stream));
        PROPAGATE_CUDA_ERROR(cudaFreeAsync(indexs, stream));
        for (int i = 0; i < max_scaler_stages; i++) {
            PROPAGATE_CUDA_ERROR(cudaFreeAsync(scalers[i], stream));
        }
        for (int i = 0; i < max_point_stages; i++) {
            PROPAGATE_CUDA_ERROR(cudaFreeAsync(points[i], stream));
        }
        PROPAGATE_CUDA_ERROR(cudaFreeAsync(d_temp_storage_sort, stream));
        for (int i = 0; i < batch_per_run; i++) {
            PROPAGATE_CUDA_ERROR(cudaFreeAsync(reduce_buffer[i], stream));
        }
        for (int i = 0; i < max_scaler_stages; i++) {
            PROPAGATE_CUDA_ERROR(cudaEventDestroy(begin_scaler_copy[i]));
            PROPAGATE_CUDA_ERROR(cudaEventDestroy(end_scaler_copy[i]));
        }
        for (int i = 0; i < max_point_stages; i++) {
            PROPAGATE_CUDA_ERROR(cudaEventDestroy(begin_point_copy[i]));
            PROPAGATE_CUDA_ERROR(cudaEventDestroy(end_point_copy[i]));
        }
        allocated = false;
        return cudaSuccess;
    }

    template <typename Config, typename Element, typename Point, typename PointAffine, typename PointAll>
    cudaError_t MSM<Config, Element, Point, PointAffine, PointAll>::msm(const std::vector<u32*>& h_scalers, std::vector<PointAll> &h_result, cudaStream_t stream) {
        // note: this function is not async, it will block until all computation is done
        // this is because the host reduce is done on the cpu
        assert(points_set);
        assert(h_scalers.size() == h_result.size());

        const u32 batches = h_scalers.size();
        cudaError_t err;
        PROPAGATE_CUDA_ERROR(cudaSetDevice(device));
        if (!allocated) PROPAGATE_CUDA_ERROR(alloc_gpu(stream));
        // TODO: use thread pool
        std::thread host_reduce_thread; // overlap host reduce with GPU computation

        auto host_reduce = [](Point **reduce_buffer, std::vector<PointAll>::iterator h_result, u32 n_reduce, u32 batches, cudaEvent_t start_reduce) {
            cudaEventSynchronize(start_reduce);
            // host timer
            std::chrono::high_resolution_clock::time_point start, end;
            if constexpr (Config::debug) {
                start = std::chrono::high_resolution_clock::now();
            }
            for (u32 j = 0; j < batches; j++, h_result++) {
                *h_result = PointAll::identity();
                for (u32 i = 0; i < n_reduce; i++) {
                    PointAll p;
                    for(u32 k = 0; k < TPI; ++k) {
                        int PER_LIMBS = Config::LIMBS / TPI;
                        for(u32 l = 0; l < PER_LIMBS; ++l) {
                            p.x.n._limbs[k*PER_LIMBS+l] = reduce_buffer[j][i*TPI+k].x.n._limbs[l];
                            p.y.n._limbs[k*PER_LIMBS+l] = reduce_buffer[j][i*TPI+k].y.n._limbs[l];
                            p.zz.n._limbs[k*PER_LIMBS+l] = reduce_buffer[j][i*TPI+k].zz.n._limbs[l];
                            p.zzz.n._limbs[k*PER_LIMBS+l] = reduce_buffer[j][i*TPI+k].zzz.n._limbs[l];
                        }
                    }
                    *h_result = *h_result + p;
                }
            }
            if constexpr (Config::debug) {
                end = std::chrono::high_resolution_clock::now();
                std::cout << "MSM host reduce time:" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << "ms" << std::endl;
            }
        };

        for (u32 i = 0; i < batches; i += batch_per_run) {
            u32 cur_batch = std::min(batch_per_run, batches - i);
            cudaEvent_t start_reduce;
            PROPAGATE_CUDA_ERROR(cudaEventCreateWithFlags(&start_reduce, cudaEventBlockingSync));
            // printf("Stage2\n");
            PROPAGATE_CUDA_ERROR(run(cur_batch, h_scalers.begin() + i, i == 0, stream));
            // printf("Stage10\n");

            if (i > 0) host_reduce_thread.join();
            
            for (int j = 0; j < cur_batch; j++) {
                PROPAGATE_CUDA_ERROR(cudaMemcpyAsync(h_reduce_buffer[j], reduce_buffer[j], sizeof(Point) * reduce_blocks * TPI, cudaMemcpyDeviceToHost, stream));
            }
            PROPAGATE_CUDA_ERROR(cudaEventRecord(start_reduce, stream));
            // printf("Stage11\n");
            host_reduce_thread = std::thread(host_reduce, h_reduce_buffer, h_result.begin() + i, reduce_blocks, cur_batch, start_reduce);
            // printf("Stage12\n");
        }

        host_reduce_thread.join();

        PROPAGATE_CUDA_ERROR(cudaGetLastError());
        // printf("Stage13\n");

        return cudaSuccess;
    }

    template <typename Config, typename Element, typename Point, typename PointAffine, typename PointAll>
    MSM<Config, Element, Point, PointAffine, PointAll>::~MSM() {
        free_gpu();
        delete [] d_buckets_sum_buf;
        delete [] buckets_sum;
        delete [] initialized_buf;
        delete [] initialized;
        delete [] scalers;
        delete [] points;
        delete [] begin_scaler_copy;
        delete [] end_scaler_copy;
        delete [] begin_point_copy;
        delete [] end_point_copy;
        delete [] reduce_buffer;
        for (int j = 0; j < batch_per_run; j++) {
            cudaFreeHost(h_reduce_buffer[j]);
        }
        delete [] h_reduce_buffer;
    }

    template <typename Config, typename Point, typename PointAffine>
    cudaError_t MSMPrecompute<Config, Point, PointAffine>::run(u64 len, u64 part_len, std::array<u32*, Config::n_precompute> h_points, cudaStream_t stream) {
        cudaError_t err;
        if constexpr (Config::n_precompute == 1) {
            return cudaSuccess;
        }
        u32 *points;
        PROPAGATE_CUDA_ERROR(cudaMallocAsync(&points, sizeof(PointAffine) * part_len * Config::n_precompute * TPI, stream));
        for (int i = 0; i * part_len < len; i++) {
            u64 offset = i * part_len;
            u64 cur_len = std::min(part_len, len - offset);

            PROPAGATE_CUDA_ERROR(cudaMemcpyAsync(points, h_points[0] + offset * PointAffine::N_WORDS * TPI, sizeof(PointAffine) * part_len * TPI, cudaMemcpyHostToDevice, stream));

            u32 grid = div_ceil(cur_len, 256) * TPI;
            u32 block = 256;
            precompute_kernel<Config, Point, PointAffine><<<grid, block, 0, stream>>>(points, cur_len * TPI);

            for (int j = 1; j < Config::n_precompute; j++) {
                PROPAGATE_CUDA_ERROR(cudaMemcpyAsync(h_points[j] + offset * PointAffine::N_WORDS * TPI, points + j * cur_len * PointAffine::N_WORDS * TPI, sizeof(PointAffine) * cur_len * TPI, cudaMemcpyDeviceToHost, stream));
            }
        }
        PROPAGATE_CUDA_ERROR(cudaFreeAsync(points, stream));
        return cudaSuccess;
    }

    // synchronous function
    template <typename Config, typename Point, typename PointAffine>
    cudaError_t MSMPrecompute<Config, Point, PointAffine>::precompute(u64 len, std::array<u32*, Config::n_precompute> h_points, int max_devices) {
        cudaError_t err;
        int devices;
        PROPAGATE_CUDA_ERROR(cudaGetDeviceCount(&devices));
        devices = std::min(devices, max_devices);
        if (Config::debug) std::cout << "Using " << devices << " devices" << std::endl;
        std::vector<cudaStream_t> streams;
        streams.resize(devices);
        u64 part_len = div_ceil(len, devices);
        for (u32 i = 0; i < devices; i++) {
            if (Config::debug) std::cout << "Precomputing on device " << i << std::endl;
            u64 offset = i * part_len;
            u64 cur_len = std::min(part_len, len - offset);
            PROPAGATE_CUDA_ERROR(cudaSetDevice(i));
            PROPAGATE_CUDA_ERROR(cudaStreamCreate(&streams[i]));
            std::array<u32*, Config::n_precompute> cur_points;
            for (int j = 0; j < Config::n_precompute; j++) {
                cur_points[j] = h_points[j] + offset * PointAffine::N_WORDS * TPI;
            }
            PROPAGATE_CUDA_ERROR(run(cur_len, std::min(cur_len, 1ul << 20), cur_points, streams[i]));
        }
        for (u32 i = 0; i < devices; i++) {
            PROPAGATE_CUDA_ERROR(cudaStreamSynchronize(streams[i]));
            PROPAGATE_CUDA_ERROR(cudaStreamDestroy(streams[i]));
        }
        return cudaSuccess;
    }

    // each msm will be decomposed into multiple msm instances, each instance will be run on a single GPU
    template <typename Config, typename Element, typename Point, typename PointAffine, typename PointAll>
    MultiGPUMSM<Config, Element, Point, PointAffine, PointAll>::MultiGPUMSM(u64 len, u32 batch_per_run, u32 parts, u32 scaler_stages, u32 point_stages, std::vector<u32> cards)
    : len(len), part_len(div_ceil(len, cards.size())), batch_per_run(batch_per_run), parts(parts),
    scaler_stages(scaler_stages), point_stages(point_stages), cards(cards) {
        msm_instances.reserve(cards.size());
        streams.resize(cards.size());
        threads.resize(cards.size());
        for (u32 i = 0; i < cards.size(); i++) {
            u64 offset = i * part_len;
            u64 cur_len = std::min(part_len, len - offset);
            msm_instances.emplace_back(cur_len, batch_per_run, parts, scaler_stages, point_stages, cards[i]);
        }
    }

    template <typename Config, typename Element, typename Point, typename PointAffine, typename PointAll>
    cudaError_t MultiGPUMSM<Config, Element, Point, PointAffine, PointAll>::alloc_gpu() {
        cudaError_t err;
        if (allocated) return cudaSuccess;
        for (u32 i = 0; i < cards.size(); i++) {
            PROPAGATE_CUDA_ERROR(cudaSetDevice(cards[i]));
            PROPAGATE_CUDA_ERROR(msm_instances[i].alloc_gpu());
            PROPAGATE_CUDA_ERROR(cudaStreamCreate(&streams[i]));
        }
        allocated = true;
        return cudaSuccess;
    }

    template <typename Config, typename Element, typename Point, typename PointAffine, typename PointAll>
    cudaError_t MultiGPUMSM<Config, Element, Point, PointAffine, PointAll>::free_gpu() {
        cudaError_t err;
        if (!allocated) return cudaSuccess;
        for (u32 i = 0; i < cards.size(); i++) {
            PROPAGATE_CUDA_ERROR(cudaSetDevice(cards[i]));
            PROPAGATE_CUDA_ERROR(msm_instances[i].free_gpu(streams[i]));
            PROPAGATE_CUDA_ERROR(cudaStreamSynchronize(streams[i]));
            PROPAGATE_CUDA_ERROR(cudaStreamDestroy(streams[i]));
        }
        allocated = false;
        return cudaSuccess;
    }

    template <typename Config, typename Element, typename Point, typename PointAffine, typename PointAll>
    void MultiGPUMSM<Config, Element, Point, PointAffine, PointAll>::set_points(std::array<u32*, Config::n_precompute> host_points) {
        for (u32 i = 0; i < cards.size(); i++) {
            u64 offset = i * part_len;
            u64 cur_len = std::min(part_len, len - offset);
            std::array<u32*, Config::n_precompute> cur_points;
            for (int j = 0; j < Config::n_precompute; j++) {
                cur_points[j] = host_points[j] + offset * PointAffine::N_WORDS * TPI;
            }
            msm_instances[i].set_points(cur_points);
        }
    }

    template <typename Config, typename Element, typename Point, typename PointAffine, typename PointAll>
    cudaError_t MultiGPUMSM<Config, Element, Point, PointAffine, PointAll>::msm(const std::vector<u32*>& h_scalers, std::vector<PointAll> &h_result) {
        assert(h_scalers.size() == h_result.size());

        cudaError_t err;
        if (!allocated) PROPAGATE_CUDA_ERROR(alloc_gpu());

        // pre-card results
        std::vector<std::vector<PointAll>> results(cards.size());
        for (u32 i = 0; i < cards.size(); i++) {
            results[i].resize(h_result.size());
        }
        
        auto run_msm = [](MSM<Config, Element, Point, PointAffine, PointAll> &msm, const std::vector<u32*> &h_scalers, std::vector<PointAll> &h_result, cudaStream_t stream) {
            msm.msm(h_scalers, h_result, stream);
        };

        for (u32 i = 0; i < cards.size(); i++) {
            u64 offset = i * part_len;
            u64 cur_len = std::min(part_len, len - offset);
            std::vector<u32*> cur_scalers = h_scalers;
            for (u32 j = 0; j < h_scalers.size(); j++) {
                cur_scalers[j] += offset * Config::LIMBS;
            }
            threads[i] = std::thread(run_msm, std::ref(msm_instances[i]), cur_scalers, std::ref(results[i]), streams[i]);
        }
        // printf("Stage14\n");

        for (u32 i = 0; i < cards.size(); i++) {
            threads[i].join();
        }

        PROPAGATE_CUDA_ERROR(cudaGetLastError());

        for (u32 i = 0; i < h_result.size(); i++) {
            h_result[i] = PointAll::identity();
            for (u32 j = 0; j < cards.size(); j++) {
                h_result[i] = h_result[i] + results[j][i];
            }
        }
        // printf("Stage15\n");
        return cudaSuccess;
    }

    template <typename Config, typename Element, typename Point, typename PointAffine, typename PointAll>
    MultiGPUMSM<Config, Element, Point, PointAffine, PointAll>::~MultiGPUMSM() {
        free_gpu();
    }
}