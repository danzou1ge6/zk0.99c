#pragma once
#include <cassert>
#include "ntt.cuh"
#include <cooperative_groups.h>

namespace ntt {


    template <typename Field>
    __global__ void ntt_kernel (u32 * x, u32 log_len, u32 log_stride, u32 deg, u32 * roots) {
        constexpr usize WORDS = Field::LIMBS;
        static_assert(WORDS % 4 == 0);
        constexpr u32 io_group = 1 << (log2_int(WORDS - 1) - 1);
        extern __shared__ u32 s[];

        const u32 lid = threadIdx.x;
        const u32 lsize = blockDim.x;
        const u32 index = blockIdx.x;

        auto u = s;

        const u32 lgp = log_stride - deg + 1;
        const u32 end_stride = 1 << lgp; //stride of the last butterfly

        // each segment is independent
        const u32 segment_id = index & (end_stride - 1);
        
        const u32 subblock_sz = 1 << (deg - 1); // # of neighbouring butterfly in the last round

        // x += ((u64)(segment_start + segment_id)) * WORDS; // use u64 to avoid overflow

        // const u32 io_id = lid & (io_group - 1);
        // const u32 lid_start = lid - io_id;
        const u32 shared_read_stride = (lsize << 1) + 1;

        const u32 tid = threadIdx.x + blockDim.x * (u32)blockIdx.x;

        const u32 inp_mask = ((u32)1 << (log_stride)) - 1;
        const u32 out_mask = end_stride - 1;

        // rearrange |tid|'s bits
        u32 idx0 = (tid & ~inp_mask) * 2;
        idx0 += (tid << lgp) & inp_mask;
        idx0 += (tid >> (deg - 1)) & out_mask;
        u32 idx1 = idx0 + ((u32)1 << (log_stride));

        auto a = reinterpret_cast<Field*>(x)[idx0];
        a.store(u + lid, shared_read_stride);
        auto b = reinterpret_cast<Field*>(x)[idx1];
        b.store(u + lid + subblock_sz, shared_read_stride);

        __syncthreads();

        const u32 pqshift = log_len - 1 - log_stride;

        u32 sub_deg = deg - 6;

        for(u32 rnd = 0; rnd < deg; rnd += sub_deg) {
            if (rnd != 0) sub_deg = 6;
            // u32 sub_deg = min(6, deg - rnd);
            u32 warp_sz = 1 << (sub_deg - 1);
            u32 warp_id = lid / warp_sz;
            
            u32 lgp = deg - rnd - sub_deg;
            u32 end_stride_warp = 1 << lgp;

            u32 segment_start_warp = (warp_id >> lgp) << (lgp + sub_deg);
            u32 segment_id_warp = warp_id & (end_stride_warp - 1);
            
            u32 laneid = lid & (warp_sz - 1);

            u32 bit = subblock_sz >> rnd;
            u32 i0 = segment_start_warp + segment_id_warp + laneid * end_stride_warp;
            u32 i1 = i0 + bit;

            a = Field::load(u + i0, shared_read_stride);
            b = Field::load(u + i1, shared_read_stride);

            for (u32 i = 0; i < sub_deg; i++) {
                if (i != 0) {
                    u32 lanemask = 1 << (sub_deg - i - 1);
                    Field tmp;
                    tmp = ((lid / lanemask) & 1) ? a : b;

                    #pragma unroll
                    for (u32 j = 0; j < WORDS; j++) {
                        tmp.n.limbs[j] = __shfl_xor_sync(0xffffffff, tmp.n.limbs[j], lanemask);
                    }

                    if ((lid / lanemask) & 1) a = tmp;
                    else b = tmp;
                }

                auto tmp = a;
                a = a + b;
                b = tmp - b;
                u32 bit = (1 << sub_deg) >> (i + 1);
                u64 di = ((lid & (bit - 1)) * end_stride_warp + segment_id_warp) * end_stride + segment_id;

                if (di != 0) {
                    auto w = Field::load(roots + (di << (rnd + i) << pqshift) * WORDS);
                    b = b * w;
                }
            }            

            i0 = segment_start_warp + segment_id_warp + laneid * 2 * end_stride_warp;
            i1 = i0 + end_stride_warp;

            if (rnd + sub_deg < deg) {
                a.store(u + i0, shared_read_stride);
                b.store(u + i1, shared_read_stride);
            }

            __syncthreads();
        }

        u32 mask = (u32)((1 << deg) - 1) << (log_stride + 1 - deg);
        u32 rotw = idx0 & mask;
        rotw = (rotw << 1) | (rotw >> (deg - 1));
        idx0 = (idx0 & ~mask) | (rotw & mask);
        rotw = idx1 & mask;
        rotw = (rotw << 1) | (rotw >> (deg - 1));
        idx1 = (idx1 & ~mask) | (rotw & mask);

        reinterpret_cast<Field*>(x)[idx0] = a;
        reinterpret_cast<Field*>(x)[idx1] = b;
    }

    template <typename Field>
    __global__ void ntt_shfl_co (u32_E * data, u32 log_len, u32 log_stride, u32 deg, u32 * roots) {
        const static usize WORDS = Field::LIMBS;
        static_assert(WORDS % 4 == 0);
        using barrier = cuda::barrier<cuda::thread_scope_block>;
        #pragma nv_diag_suppress static_var_with_dynamic_init
        __shared__  barrier bar;

        if (threadIdx.x == 0) {
            init(&bar, blockDim.x); // Initialize the barrier with expected arrival count
        }
        __syncthreads();

        extern __shared__ uint4 shfl[];

        const u32 lid = threadIdx.x;
        const u32 lsize = blockDim.x;
        const u32 index = blockIdx.x;

        const u32 lgp = log_stride - deg + 1;
        const u32 end_stride = 1 << lgp; //stride of the last butterfly

        // each segment is independent
        const u32 segment_start = (index >> lgp) << (lgp + deg);
        const u32 segment_id = index & (end_stride - 1);
        
        const u32 subblock_sz = 1 << (deg - 1); // # of neighbouring butterfly in the last round

        data += ((u64)(segment_start + segment_id)) * WORDS; // use u64 to avoid overflow
        
        Field a, b;

        auto tile = cooperative_groups::tiled_partition<2>(cooperative_groups::this_thread_block());

        // Read data
        if (WORDS == 8) {
            u64 m_gpos = ((tile.meta_group_rank() * 2) & (subblock_sz - 1)) << lgp;
            u64 n_gpos = ((tile.meta_group_rank() * 2 + 1) & (subblock_sz - 1)) << lgp;

            reinterpret_cast<uint4*>(a.n.limbs)[0] = reinterpret_cast<uint4*>(data + m_gpos * WORDS)[tile.thread_rank()];
            reinterpret_cast<uint4*>(a.n.limbs)[1] = reinterpret_cast<uint4*>(data + n_gpos * WORDS)[tile.thread_rank()];

            reinterpret_cast<uint4*>(b.n.limbs)[0] = reinterpret_cast<uint4*>(data + (m_gpos + (end_stride << (deg - 1))) * WORDS)[tile.thread_rank()];
            reinterpret_cast<uint4*>(b.n.limbs)[1] = reinterpret_cast<uint4*>(data + (n_gpos + (end_stride << (deg - 1))) * WORDS)[tile.thread_rank()];

            uint4 shfla = tile.thread_rank() ? reinterpret_cast<uint4*>(a.n.limbs)[0] : reinterpret_cast<uint4*>(a.n.limbs)[1];
            uint4 shflb = tile.thread_rank() ? reinterpret_cast<uint4*>(b.n.limbs)[0] : reinterpret_cast<uint4*>(b.n.limbs)[1];

            shfla.x = tile.shfl_xor(shfla.x, 1), shfla.y = tile.shfl_xor(shfla.y, 1), shfla.z = tile.shfl_xor(shfla.z, 1), shfla.w = tile.shfl_xor(shfla.w, 1);
            shflb.x = tile.shfl_xor(shflb.x, 1), shflb.y = tile.shfl_xor(shflb.y, 1), shflb.z = tile.shfl_xor(shflb.z, 1), shflb.w = tile.shfl_xor(shflb.w, 1);

            if (tile.thread_rank()) {
                reinterpret_cast<uint4*>(a.n.limbs)[0] = shfla;
                reinterpret_cast<uint4*>(b.n.limbs)[0] = shflb;
            } else {
                reinterpret_cast<uint4*>(a.n.limbs)[1] = shfla;
                reinterpret_cast<uint4*>(b.n.limbs)[1] = shflb;
            }
        } else {
            u64 gpos = (lid & (subblock_sz - 1)) << lgp;

            a = Field::load(data + (gpos) * WORDS);
            b = Field::load(data + (gpos + (end_stride << (deg - 1))) * WORDS);
        }


        barrier::arrival_token token = bar.arrive(); /* this thread arrives. Arrival does not block a thread */

        const u32 pqshift = log_len - 1 - log_stride;

        u32 i = 0;
        for (; i + 6 < deg; i++) {
            auto tmp = a;
            a = a + b;
            b = tmp - b;

            u32 bit = subblock_sz >> i;
            u64 di = (lid & (bit - 1)) * end_stride + segment_id;

            if (di != 0) {
                auto w = reinterpret_cast<Field*>(roots)[(di << i << pqshift)];
                b = b * w;
            }
            
            u32 lanemask = 1 << (deg - i - 2);
            tmp = ((lid / lanemask) & 1) ? a : b;
            #pragma unroll
            for (u32 j = 0; j < WORDS; j += 4) {
                uint4 seg = uint4 {tmp.n.limbs[j], tmp.n.limbs[j + 1], tmp.n.limbs[j + 2], tmp.n.limbs[j + 3]};
                u32 offset = j / 4 * blockDim.x + lid;
                shfl[offset] = seg;
            }
            __syncthreads();
            #pragma unroll
            for (u32 j = 0; j < WORDS; j += 4) {
                u32 offset = j / 4 * blockDim.x + (lid ^ lanemask);
                uint4 seg = shfl[offset];
                tmp.n.limbs[j] = seg.x;
                tmp.n.limbs[j + 1] = seg.y;
                tmp.n.limbs[j + 2] = seg.z;
                tmp.n.limbs[j + 3] = seg.w;
            }
            if ((lid / lanemask) & 1) a = tmp;
            else b = tmp;
        }

        for (; i < deg; i++) {
            auto tmp = a;
            a = a + b;
            b = tmp - b;

            u32 bit = subblock_sz >> i;
            u64 di = (lid & (bit - 1)) * end_stride + segment_id;

            if (di != 0) {
                auto w = reinterpret_cast<Field*>(roots)[(di << i << pqshift)];
                b = b * w;
            }
            
            if (i + 1 < deg) {
                u32 lanemask = 1 << (deg - i - 2);
                tmp = ((lid / lanemask) & 1) ? a : b;

                #pragma unroll
                for (u32 j = 0; j < WORDS; j++) {
                    tmp.n.limbs[j] = __shfl_xor_sync(0xffffffff, tmp.n.limbs[j], lanemask);
                }

                if ((lid / lanemask) & 1) a = tmp;
                else b = tmp;
            }
        }
        
        bar.wait(std::move(token)); /* wait for all threads participating in the barrier to complete bar.arrive()*/

        // Write back
        if (WORDS == 8) {
            u64 m_gpos = ((tile.meta_group_rank() * 2) & (subblock_sz - 1)) << (lgp + 1);
            u64 n_gpos = ((tile.meta_group_rank() * 2 + 1) & (subblock_sz - 1)) << (lgp + 1);
            
            uint4 shfla = tile.thread_rank() ? reinterpret_cast<uint4*>(a.n.limbs)[0] : reinterpret_cast<uint4*>(a.n.limbs)[1];
            uint4 shflb = tile.thread_rank() ? reinterpret_cast<uint4*>(b.n.limbs)[0] : reinterpret_cast<uint4*>(b.n.limbs)[1];

            shfla.x = tile.shfl_xor(shfla.x, 1), shfla.y = tile.shfl_xor(shfla.y, 1), shfla.z = tile.shfl_xor(shfla.z, 1), shfla.w = tile.shfl_xor(shfla.w, 1);
            shflb.x = tile.shfl_xor(shflb.x, 1), shflb.y = tile.shfl_xor(shflb.y, 1), shflb.z = tile.shfl_xor(shflb.z, 1), shflb.w = tile.shfl_xor(shflb.w, 1);
            
            if (tile.thread_rank()) {
                reinterpret_cast<uint4*>(a.n.limbs)[0] = shfla;
                reinterpret_cast<uint4*>(b.n.limbs)[0] = shflb;
            } else {
                reinterpret_cast<uint4*>(a.n.limbs)[1] = shfla;
                reinterpret_cast<uint4*>(b.n.limbs)[1] = shflb;
            }

            reinterpret_cast<uint4*>(data + m_gpos * WORDS)[tile.thread_rank()] = reinterpret_cast<uint4*>(a.n.limbs)[0];
            reinterpret_cast<uint4*>(data + n_gpos * WORDS)[tile.thread_rank()] = reinterpret_cast<uint4*>(a.n.limbs)[1];

            reinterpret_cast<uint4*>(data + (m_gpos + end_stride) * WORDS)[tile.thread_rank()] = reinterpret_cast<uint4*>(b.n.limbs)[0];
            reinterpret_cast<uint4*>(data + (n_gpos + end_stride) * WORDS)[tile.thread_rank()] = reinterpret_cast<uint4*>(b.n.limbs)[1];
        } else {
            u32 group_id = lid & (subblock_sz - 1);
            u64 gpos = group_id << (lgp + 1);
            
            a.store(data + (gpos) * WORDS);
            b.store(data + (gpos + end_stride) * WORDS);
        }
    }

    template <typename Field>
    __global__ void ntt_shared_shfl (u32 * x, u32 log_len, u32 log_stride, u32 deg, u32 * roots) {
        constexpr usize WORDS = Field::LIMBS;
        static_assert(WORDS % 4 == 0);
        constexpr u32 io_group = 1 << (log2_int(WORDS - 1) - 1);
        extern __shared__ uint4 shfl[];

        const u32 lid = threadIdx.x;
        const u32 lsize = blockDim.x;
        const u32 index = blockIdx.x;

        const u32 lgp = log_stride - deg + 1;
        const u32 end_stride = 1 << lgp; //stride of the last butterfly

        // each segment is independent
        const u32 segment_id = index & (end_stride - 1);
        
        const u32 subblock_sz = 1 << (deg - 1); // # of neighbouring butterfly in the last round

        // x += ((u64)(segment_start + segment_id)) * WORDS; // use u64 to avoid overflow

        // const u32 io_id = lid & (io_group - 1);
        // const u32 lid_start = lid - io_id;
        const u32 shared_read_stride = (lsize << 1) + 1;

        const u32 tid = threadIdx.x + blockDim.x * (u32)blockIdx.x;

        const u32 inp_mask = ((u32)1 << (log_stride)) - 1;
        const u32 out_mask = end_stride - 1;

        // rearrange |tid|'s bits
        u32 idx0 = (tid & ~inp_mask) * 2;
        idx0 += (tid << lgp) & inp_mask;
        idx0 += (tid >> (deg - 1)) & out_mask;
        u32 idx1 = idx0 + ((u32)1 << (log_stride));

        auto a = reinterpret_cast<Field*>(x)[idx0];
        auto b = reinterpret_cast<Field*>(x)[idx1];

        const u32 pqshift = log_len - 1 - log_stride;

        u32 i = 0;
        for (; i + 6 < deg; i++) {
            auto tmp = a;
            a = a + b;
            b = tmp - b;

            u32 bit = subblock_sz >> i;
            u64 di = (lid & (bit - 1)) * end_stride + segment_id;

            if (di != 0) {
                auto w = reinterpret_cast<Field*>(roots)[(di << i << pqshift)];
                b = b * w;
            }
            
            u32 lanemask = 1 << (deg - i - 2);
            tmp = ((lid / lanemask) & 1) ? a : b;
            #pragma unroll
            for (u32 j = 0; j < WORDS; j += 4) {
                uint4 seg = uint4 {tmp.n.limbs[j], tmp.n.limbs[j + 1], tmp.n.limbs[j + 2], tmp.n.limbs[j + 3]};
                u32 offset = j / 4 * blockDim.x + lid;
                shfl[offset] = seg;
            }
            __syncthreads();
            #pragma unroll
            for (u32 j = 0; j < WORDS; j += 4) {
                u32 offset = j / 4 * blockDim.x + (lid ^ lanemask);
                uint4 seg = shfl[offset];
                tmp.n.limbs[j] = seg.x;
                tmp.n.limbs[j + 1] = seg.y;
                tmp.n.limbs[j + 2] = seg.z;
                tmp.n.limbs[j + 3] = seg.w;
            }
            if ((lid / lanemask) & 1) a = tmp;
            else b = tmp;
        }

        for (; i < deg; i++) {
            auto tmp = a;
            a = a + b;
            b = tmp - b;

            u32 bit = subblock_sz >> i;
            u64 di = (lid & (bit - 1)) * end_stride + segment_id;

            if (di != 0) {
                auto w = reinterpret_cast<Field*>(roots)[(di << i << pqshift)];
                b = b * w;
            }
            
            if (i + 1 < deg) {
                u32 lanemask = 1 << (deg - i - 2);
                tmp = ((lid / lanemask) & 1) ? a : b;

                #pragma unroll
                for (u32 j = 0; j < WORDS; j++) {
                    tmp.n.limbs[j] = __shfl_xor_sync(0xffffffff, tmp.n.limbs[j], lanemask);
                }

                if ((lid / lanemask) & 1) a = tmp;
                else b = tmp;
            }
        }

        u32 mask = (u32)((1 << deg) - 1) << (log_stride + 1 - deg);
        u32 rotw = idx0 & mask;
        rotw = (rotw << 1) | (rotw >> (deg - 1));
        idx0 = (idx0 & ~mask) | (rotw & mask);
        rotw = idx1 & mask;
        rotw = (rotw << 1) | (rotw >> (deg - 1));
        idx1 = (idx1 & ~mask) | (rotw & mask);

        reinterpret_cast<Field*>(x)[idx0] = a;
        reinterpret_cast<Field*>(x)[idx1] = b;
    }

    template <typename Field>
    class cooley_turkey_ntt : public best_ntt {
        const static usize WORDS = Field::LIMBS;
        using Number = mont::Number<WORDS>;
        
        u32 max_deg;
        const int log_len;
        const u64 len;
        Field unit;
        bool debug;
        u32 *roots, *roots_d;
        const bool inverse;
        const bool process;
        u32 * inv_n, * inv_n_d;
        u32 * zeta, * zeta_d;

        u32 get_deg (u32 deg_stage, u32 max_deg_stage) {
            u32 deg_per_round;
            for (u32 rounds = 1; ; rounds++) {
                deg_per_round = rounds == 1 ? deg_stage : (deg_stage - 1) / rounds + 1;
                if (deg_per_round <= max_deg_stage) break;
            }
            return deg_per_round;
        }

        public:            
        u32 max_threads_stage_log = 10;
        float milliseconds = 0;

        cooley_turkey_ntt(
            const u32* omega, u32 log_len, 
            bool debug, 
            u32 max_instance = 1,
            bool inverse = false, 
            bool process = false, 
            const u32 * inv_n = nullptr, 
            const u32 * zeta = nullptr) 
        : best_ntt(max_instance), log_len(log_len), len(1ll << log_len), debug(debug)
        , inverse(inverse), process(process){
            cudaError_t first_err = cudaSuccess;

            max_deg = get_deg(log_len, max_threads_stage_log + 1);

            // Precalculate:
            if (debug) {
                // unit = qpow(omega, (P - 1ll) / len)
                unit = Field::from_number(Number::load(omega));
                auto one = Number::zero();
                one.limbs[0] = 1;
                Number exponent = (Field::ParamsType::m() - one).slr(log_len);
                unit = unit.pow(exponent);
            } else {
                unit = Field::load(omega);
            }

            
            CUDA_CHECK(cudaHostAlloc(&roots, (u64)len / 2 * WORDS * sizeof(u32), cudaHostAllocDefault));
            gen_roots_cub<Field> gen;
            CUDA_CHECK(gen(roots, len / 2, unit));

            if (inverse) {
                assert(inv_n != nullptr);
                CUDA_CHECK(cudaHostAlloc(&this->inv_n, sizeof(u32) * WORDS, cudaHostAllocDefault));
                CUDA_CHECK(cudaMemcpy(this->inv_n, inv_n, sizeof(u32) * WORDS, cudaMemcpyHostToHost));
            }

            if (process) {
                assert(zeta != nullptr);

                CUDA_CHECK(cudaHostAlloc(&this->zeta, 2 * WORDS * sizeof(u32),cudaHostAllocDefault));
                CUDA_CHECK(cudaMemcpy(this->zeta, zeta, 2 * WORDS * sizeof(u32), cudaMemcpyHostToDevice));
            }

            if (first_err != cudaSuccess) {
                std::cerr << "error occurred during gen_roots" << std::endl;
                throw cudaGetErrorString(first_err);
            }
        }

        ~cooley_turkey_ntt() override {
            cudaFreeHost(roots);
            if (inverse) cudaFreeHost(inv_n);
            if (process) cudaFreeHost(zeta);
            if (on_gpu) clean_gpu();
        }

        cudaError_t to_gpu(cudaStream_t stream = 0) override {
            std::unique_lock<std::shared_mutex> wlock(this->mtx);
            cudaError_t first_err = cudaSuccess;

            CUDA_CHECK(cudaMallocAsync(&roots_d, len / 2 * WORDS * sizeof(u32), stream));
            CUDA_CHECK(cudaMemcpyAsync(roots_d, roots, len / 2 * WORDS * sizeof(u32), cudaMemcpyHostToDevice, stream));

            if (inverse) {
                CUDA_CHECK(cudaMallocAsync(&inv_n_d, WORDS * sizeof(u32), stream));
                CUDA_CHECK(cudaMemcpyAsync(inv_n_d, inv_n, WORDS * sizeof(u32), cudaMemcpyHostToDevice, stream));
            }

            if (process) {
                CUDA_CHECK(cudaMallocAsync(&zeta_d, 2 * WORDS * sizeof(u32), stream));
                CUDA_CHECK(cudaMemcpyAsync(zeta_d, zeta, 2 * WORDS * sizeof(u32), cudaMemcpyHostToDevice, stream));
            }

            if (first_err != cudaSuccess) {
                CUDA_CHECK(cudaFreeAsync(roots_d, stream));
                if (inverse) CUDA_CHECK(cudaFreeAsync(inv_n_d, stream));
                if (process) CUDA_CHECK(cudaFreeAsync(zeta_d, stream));
            } else {
                this->on_gpu = true;
            }
            return first_err;
        }

        cudaError_t clean_gpu(cudaStream_t stream = 0) override {
            std::unique_lock<std::shared_mutex> wlock(this->mtx);
            if (!this->on_gpu) return cudaSuccess;
            cudaError_t first_err = cudaSuccess;

            CUDA_CHECK(cudaFreeAsync(roots_d, stream));

            if (inverse) CUDA_CHECK(cudaFreeAsync(inv_n_d, stream));
            if (process) CUDA_CHECK(cudaFreeAsync(zeta_d, stream));

            this->on_gpu = false;
            return first_err;
        }

        cudaError_t ntt(u32 * data, cudaStream_t stream = 0, u32 start_n = 0, bool data_on_gpu = false) override {
            cudaError_t first_err = cudaSuccess;

            if (log_len == 0) return first_err;

            cudaEvent_t start, end;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&end));
            
            this->sem.acquire();

            std::shared_lock<std::shared_mutex> rlock(this->mtx);
            {
                while(!this->on_gpu) {
                    rlock.unlock();
                    CUDA_CHECK(to_gpu(stream));
                    rlock.lock();
                }
            }

            u32 * x;
            if (data_on_gpu) {
                x = data;
            } else {
                CUDA_CHECK(cudaMallocAsync(&x, len * WORDS * sizeof(u32), stream));
                if (process && !inverse) {
                    CUDA_CHECK(cudaMemcpyAsync(x, data, start_n * WORDS * sizeof(u32), cudaMemcpyHostToDevice, stream));
                } else {
                    CUDA_CHECK(cudaMemcpyAsync(x, data, len * WORDS * sizeof(u32), cudaMemcpyHostToDevice, stream));
                }
            }

            if (debug) {
                dim3 block(768);
                dim3 grid((len - 1) / block.x + 1);
                number_to_element <Field> <<< grid, block, 0, stream >>> (x, len);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventRecord(start));
            }

            int log_stride = log_len - 1;
            constexpr u32 io_group = 1 << (log2_int(WORDS - 1) + 1);
            
            while (log_stride >= 0) {
                u32 deg = std::min((int)max_deg, log_stride + 1);

                u32 group_num = 1;

                u32 block_sz = (1 << (deg - 1)) * group_num;
                assert(block_sz <= (1 << max_threads_stage_log));
                u32 block_num = len / 2 / block_sz;
                assert(block_num * 2 * block_sz == len);

                dim3 block(block_sz);
                dim3 grid(block_num);

                auto kernel = ntt_shfl_co<Field>;// SSIP_NTT_stage1_warp_no_twiddle <Field, false>;

                u32 shared_size = sizeof(Field)  * block_sz;
                CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size));

                // kernel <<< grid, block, shared_size, stream >>>(x, log_len, log_stride, deg, 1 << (deg - 1), roots_d, zeta_d, start_n);

                kernel <<< grid, block, shared_size, stream >>>(x, log_len, log_stride, deg, roots_d);
                        
                CUDA_CHECK(cudaGetLastError());

                log_stride -= deg;
            }

            dim3 block(1024);
            dim3 grid(std::min(1024ul, (((u64) len) * WORDS / 4 - 1) / block.x + 1));
            rearrange<WORDS / 4> <<<grid, block >>> (x, log_len);

            // // manually tackle log_len == 1 case because the log_stride + 12 kernel won't run
            // if (log_len == 1) {
            //     post_process_len_2<Field><<< 1, 1, 0, stream >>>(x, inverse, process, inv_n_d, zeta_d);
            // }

            if (debug) {
                CUDA_CHECK(cudaEventRecord(end));
                CUDA_CHECK(cudaEventSynchronize(end));
                CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
                dim3 block(768);
                dim3 grid((len - 1) / block.x + 1);
                element_to_number <Field> <<< grid, block, 0, stream >>> (x, len);
                CUDA_CHECK(cudaGetLastError());
            }

            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(end));

            rlock.unlock();

            if (first_err == cudaSuccess && !data_on_gpu) CUDA_CHECK(cudaMemcpyAsync(data, x, len * WORDS * sizeof(u32), cudaMemcpyDeviceToHost, stream));

            if (!data_on_gpu) CUDA_CHECK(cudaFreeAsync(x, stream));

            if (!data_on_gpu) CUDA_CHECK(cudaStreamSynchronize(stream));

            this->sem.release();

            if (debug) CUDA_CHECK(clean_gpu(stream));

            return first_err;
        }
    };
}