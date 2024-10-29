#pragma once

#include <bit>
#include "ntt.cuh"
#include <cassert>

namespace ntt {

    template <typename Field, u32 io_group>
    __global__ void SSIP_NTT_stage1_warp_recompute (u32_E * x, const u32_E * pq, u32 log_len, u32 log_stride, u32 deg, u32 max_deg, u32 group_sz, const u32_E * omegas) {
        const static usize WORDS = Field::LIMBS;
        extern __shared__ u32_E s[];

        const u32 lid = threadIdx.x & (group_sz - 1);
        const u32 lsize = group_sz;
        const u32 group_id = threadIdx.x / group_sz;
        const u32 group_num = blockDim.x / group_sz;
        const u32 index = blockIdx.x * group_num + group_id;

        auto u = s + group_id * ((1 << deg) + 1) * WORDS;

        const u32 lgp = log_stride - deg + 1;
        const u32 end_stride = 1 << lgp; //stride of the last butterfly

        // each segment is independent
        const u32 segment_start = (index >> lgp) << (lgp + deg);
        const u32 segment_id = index & (end_stride - 1);
        
        const u32 subblock_sz = 1 << (deg - 1); // # of neighbouring butterfly in the last round

        x += ((u64)(segment_start + segment_id)) * WORDS; // use u64 to avoid overflow

        const u32 io_id = lid & (io_group - 1);
        const u32 lid_start = lid - io_id;
        const u32 shared_read_stride = (lsize << 1) + 1;
        const u32 cur_io_group = io_group < lsize ? io_group : lsize;
        const u32 io_per_thread = io_group / cur_io_group;

        const u32 io_group_id = lid / cur_io_group;
        int io_st, io_ed, io_stride;

        if ((io_group_id / 2) & 1) {
            io_st = lid_start;
            io_ed = lid_start + cur_io_group;
            io_stride = 1;
        } else {
            io_st = lid_start + cur_io_group - 1;
            io_ed = ((int)lid_start) - 1;
            io_stride = -1;
        }
        // Read data
        for (int i = io_st; i != io_ed; i += io_stride) {
            for (u32 j = 0; j < io_per_thread; j++) {
                u32 io = io_id + j * cur_io_group;
                if (io < WORDS) {
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_id << (lgp + 1);
                    u[(i << 1) + io * shared_read_stride] = x[gpos * WORDS + io];
                    u[(i << 1) + 1 + io * shared_read_stride] = x[(gpos + end_stride) * WORDS + io];
                }
            }
        }

        __syncthreads();

        const u32 pqshift = max_deg - deg;
        for(u32 rnd = 0; rnd < deg; rnd += 6) {
            u32 sub_deg = min(6, deg - rnd);
            u32 warp_sz = 1 << (sub_deg - 1);
            u32 warp_id = lid / warp_sz;
            
            u32 lgp = deg - rnd - sub_deg;
            u32 end_stride = 1 << lgp;

            u32 segment_start = (warp_id >> lgp) << (lgp + sub_deg);
            u32 segment_id = warp_id & (end_stride - 1);
            
            u32 laneid = lid & (warp_sz - 1);

            u32 bit = subblock_sz >> rnd;
            u32 i0 = segment_start + segment_id + laneid * end_stride;
            u32 i1 = i0 + bit;

            auto a = Field::load(u + i0, shared_read_stride);
            auto b = Field::load(u + i1, shared_read_stride);

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
                u32 di = (lid & (bit - 1)) * end_stride + segment_id;

                if (di != 0) {
                    auto w = Field::load(pq + (di << (rnd + i) << pqshift) * WORDS);
                    b = b * w;
                }
            }            

            i0 = segment_start + segment_id + laneid * 2 * end_stride;
            i1 = i0 + end_stride;
            a.store(u + i0, shared_read_stride);
            b.store(u + i1, shared_read_stride);

            __syncthreads();
        }

        // Twiddle factor
        u32 k = index & (end_stride - 1);
        u32 pos1 = __brev(lid << 1) >> (32 - deg);
        u32 pos2 = __brev((lid << 1) + 1) >> (32 - deg);

        auto twiddle = pow_lookup_constant <Field> (((1 << log_len) >> lgp >> deg) * k, omegas);
        auto t1 = twiddle.pow(lid << 1, deg);

        auto a = Field::load(u + pos1, shared_read_stride);
        a = a * t1;
        a.store(u + pos1, shared_read_stride);

        t1 = t1 * twiddle;

        a = Field::load(u + pos2, shared_read_stride);
        a = a * t1;
        a.store(u + pos2, shared_read_stride);

        __syncthreads();

        // Write back
        for (int i = io_st; i != io_ed; i += io_stride) {
            for (u32 j = 0; j < io_per_thread; j++) {
                u32 io = io_id + j * cur_io_group;
                if (io < WORDS) {
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_id << (lgp + 1);
                    x[gpos * WORDS + io] = u[(i << 1) + io * shared_read_stride];
                    x[(gpos + end_stride) * WORDS + io] = u[(i << 1) + 1 + io * shared_read_stride];
                }
            }
        }
    }

    template <typename Field, u32 io_group>
    __global__ void SSIP_NTT_stage2_warp_recompute (u32_E * data, const u32_E * pq, u32 log_len, u32 log_stride, u32 deg, u32 max_deg, u32 group_sz, const u32_E * omegas) {
        const static usize WORDS = Field::LIMBS;
        extern __shared__ u32_E s[];

        const u32 lid = threadIdx.x & (group_sz - 1);
        const u32 lsize = group_sz;
        const u32 group_id = threadIdx.x / group_sz;
        const u32 group_num = blockDim.x / group_sz;
        const u32 index = blockIdx.x * group_num + group_id;

        u32 log_end_stride = (log_stride - deg + 1);
        u32 end_stride = 1 << log_end_stride; //stride of the last butterfly
        u32 end_pair_stride = 1 << (log_len - log_stride - 2 + deg); // the stride between the last pair of butterfly

        // each segment is independent
        // uint segment_stride = end_pair_stride << 1; // the distance between two segment
        u32 log_segment_num = (log_len - log_stride - 1 - deg); // log of # of blocks in a segment
        
        u32 segment_start = (index >> log_segment_num) << (log_segment_num + (deg << 1)); // segment_start = index / segment_num * segment_stride;

        u32 segment_id = index & ((1 << log_segment_num) - 1); // segment_id = index & (segment_num - 1);
        
        u32 subblock_sz = 1 << (deg - 1); // # of neighbouring butterfly in the last round
        u32 subblock_offset = (segment_id >> log_end_stride) << (deg + log_end_stride); // subblock_offset = (segment_id / (end_stride)) * (2 * subblock_sz * end_stride);
        u32 subblock_id = segment_id & (end_stride - 1);

        data += ((u64)(segment_start + subblock_offset + subblock_id)) * WORDS; // use u64 to avoid overflow
        auto u = s + group_id * ((1 << (deg << 1)) + 1) * WORDS;

        const u32 io_id = lid & (io_group - 1);
        const u32 lid_start = lid - io_id;
        const u32 shared_read_stride = (lsize << 2) + 1;
        const u32 cur_io_group = io_group < lsize ? io_group : lsize;
        const u32 io_per_thread = io_group / cur_io_group;

        const u32 io_group_id = lid / cur_io_group;
        int io_st, io_ed, io_stride;

        if ((io_group_id / 2) & 1) {
            io_st = lid_start;
            io_ed = lid_start + cur_io_group;
            io_stride = 1;
        } else {
            io_st = lid_start + cur_io_group - 1;
            io_ed = ((int)lid_start) - 1;
            io_stride = -1;
        }

        // Read data
        for (int i = io_st; i != io_ed; i += io_stride) {
            for (u32 j = 0; j < io_per_thread; j++) {
                u32 io = io_id + j * cur_io_group;
                if (io < WORDS) {
                    u32 group_offset = (i >> (deg - 1)) << (log_len - log_stride - 1);
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_offset + (group_id << (log_end_stride + 1)); // group_offset + group_id * (end_stride << 1)
                    u32 offset = io * shared_read_stride;

                    u[(i << 1) + offset] = data[gpos * WORDS + io];
                    u[(i << 1) + 1 + offset] = data[(gpos + end_stride) * WORDS + io];
                    u[(i << 1) + (lsize << 1) + offset] = data[(gpos + end_pair_stride) * WORDS + io];
                    u[(i << 1) + (lsize << 1) + 1 + offset] = data[(gpos + end_pair_stride + end_stride) * WORDS + io];

                }
            }
        }

        __syncthreads();

        const u32 pqshift = max_deg - deg;
        for(u32 rnd = 0; rnd < deg; rnd += 6) {
            u32 bit = subblock_sz >> rnd;
            u32 di = lid & (bit - 1);
            u32 i0 = (lid << 1) - di;
            u32 i1 = i0 + bit;
            u32 i2 = i0 + (lsize << 1);
            u32 i3 = i2 + bit;

            auto a = Field::load(u + i0, shared_read_stride);
            auto b = Field::load(u + i1, shared_read_stride);
            auto c = Field::load(u + i2, shared_read_stride);
            auto d = Field::load(u + i3, shared_read_stride);

            u32 sub_deg = min(6, deg - rnd);

            for (u32 i = 0; i < sub_deg; i++) {
                if (i != 0) {
                    u32 lanemask = 1 << (sub_deg - i - 1);
                    Field tmp, tmp1;
                    tmp = ((lid / lanemask) & 1) ? a : b;
                    tmp1 = ((lid / lanemask) & 1) ? c : d;
                    #pragma unroll
                    for (u32 j = 0; j < WORDS; j++) {
                        tmp.n.limbs[j] = __shfl_xor_sync(0xffffffff, tmp.n.limbs[j], lanemask);
                    }

                    #pragma unroll
                    for (u32 j = 0; j < WORDS; j++) {
                        tmp1.n.limbs[j] = __shfl_xor_sync(0xffffffff, tmp1.n.limbs[j], lanemask);
                    }

                    if ((lid / lanemask) & 1) a = tmp, c = tmp1;
                    else b = tmp, d = tmp1;
                }

                auto tmp1 = a;
                auto tmp2 = c;

                a = a + b;
                c = c + d;
                b = tmp1 - b;
                d = tmp2 - d;

                bit = subblock_sz >> (rnd + i);
                di = lid & (bit - 1);

                if (di != 0) {
                    auto w = Field::load(pq + (di << (rnd + i) << pqshift) * WORDS);
                    b = b * w;
                    d = d * w;
                }
            }

            i0 = (lid << 1) - di;
            i1 = i0 + bit;
            i2 = i0 + (lsize << 1);
            i3 = i2 + bit;

            a.store(u + i0, shared_read_stride);
            b.store(u + i1, shared_read_stride);
            c.store(u + i2, shared_read_stride);
            d.store(u + i3, shared_read_stride);

            __syncthreads();
        }

        // Twiddle factor
        u32 k = index & (end_stride - 1);
        u32 n = 1 << log_len;

        auto twiddle = pow_lookup_constant<Field>((n >> log_end_stride >> deg) * k, omegas);
        auto twiddle_gap = pow_lookup_constant<Field>((n >> log_end_stride >> deg) * k * (1 << (deg - 1)), omegas);
        auto t1 = twiddle.pow(lid << 1 >> deg, deg);
        auto t2 = t1 * twiddle_gap; // env.pow(twiddle, ((lid << 1) >> deg) + ((lsize <<1) >> deg), deg);
        
        u32 a, b, c, d;
        a = __brev(lid << 1) >> (32 - (deg << 1));
        b = __brev((lid << 1) + 1) >> (32 - (deg << 1));
        c = __brev((lid << 1) + (lsize << 1)) >> (32 - (deg << 1));
        d = __brev((lid << 1) + (lsize << 1) + 1) >> (32 - (deg << 1));

        auto num = Field::load(u + a, shared_read_stride);
        num = num * t1;
        num.store(u + a, shared_read_stride);

        num = Field::load(u + b, shared_read_stride);
        num = num * t1;
        num.store(u + b, shared_read_stride);

        num = Field::load(u + c, shared_read_stride);
        num = num * t2;
        num.store(u + c, shared_read_stride);

        num = Field::load(u + d, shared_read_stride);
        num = num * t2;
        num.store(u + d, shared_read_stride);

        __syncthreads();

        // Write back
        for (int i = io_st; i != io_ed; i += io_stride) {
            for (u32 j = 0; j < io_per_thread; j++) {
                u32 io = io_id + j * cur_io_group;
                if (io < WORDS) {
                    u32 group_offset = (i >> (deg - 1)) << (log_len - log_stride - 1);
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_offset + (group_id << (log_end_stride + 1)); // group_offset + group_id * (end_stride << 1)
                    u32 offset = io * shared_read_stride;
                    a = __brev(i << 1) >> (32 - (deg << 1));
                    b = __brev((i << 1) + 1) >> (32 - (deg << 1));
                    c = __brev((i << 1) + (lsize << 1)) >> (32 - (deg << 1));
                    d = __brev((i << 1) + (lsize << 1) + 1) >> (32 - (deg << 1));

                    data[gpos * WORDS + io] = u[a + offset];
                    data[(gpos + end_stride) * WORDS + io] = u[b + offset];
                    data[(gpos + end_pair_stride) * WORDS + io] = u[c + offset];
                    data[(gpos + end_pair_stride + end_stride) * WORDS + io] = u[d + offset];

                }
            }
        }
    }


    template <typename Field>
    class recompute_ntt : public best_ntt {
        const static usize WORDS = Field::LIMBS;
        using Number = mont::Number<WORDS>;
        
        u32 max_deg_stage1;
        u32 max_deg_stage2;
        u32 max_deg;
        const int log_len;
        const u64 len;
        Field unit;
        bool debug;
        u32_E *pq = nullptr, *pq_d; // Precalculated values for radix degrees up to `max_deg`
        u32_E *omegas, *omegas_d;
        const bool inverse;
        const bool process;
        u32 * inv_n, * inv_n_d;
        u32 * zeta, * zeta_d;
        u32 max_threads_stage1_log = 8;
        u32 max_threads_stage2_log = 7;

        u32 get_deg (u32 deg_stage, u32 max_deg_stage) {
            u32 deg_per_round;
            for (u32 rounds = 1; ; rounds++) {
                deg_per_round = rounds == 1 ? deg_stage : (deg_stage - 1) / rounds + 1;
                if (deg_per_round <= max_deg_stage) break;
            }
            return deg_per_round;
        }

        static constexpr u32 log2_int(u32 x) {
            return 31 - std::countl_zero(x);
        }

        public:
        float milliseconds = 0;

        recompute_ntt(
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

            u32 deg_stage1 = (log_len + 1) / 2;
            u32 deg_stage2 = log_len / 2;

            max_deg_stage1 = get_deg(deg_stage1, max_threads_stage1_log + 1);

            max_deg_stage2 = get_deg(deg_stage2, (max_threads_stage2_log + 2) / 2); // 4 elements per thread

            max_deg = std::max(max_deg_stage1, max_deg_stage2);

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

            // pq: [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]
            CUDA_CHECK(cudaHostAlloc(&pq, (1 << max_deg >> 1) * sizeof(u32) * WORDS, cudaHostAllocDefault));
            memset(pq, 0, (1 << max_deg >> 1) * sizeof(u32) * WORDS);
            Field::one().store(pq);
            auto twiddle = unit.pow(len >> max_deg);
            if (max_deg > 1) {
                twiddle.store(pq + WORDS);
                auto last = twiddle;
                for (u64 i = 2; i < (1 << max_deg >> 1); i++) {
                    last = last * twiddle;
                    last.store(pq + i * WORDS);
                }
            }

            // omegas: [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]

            CUDA_CHECK(cudaHostAlloc(&omegas, 32 * sizeof(u32) * WORDS, cudaHostAllocDefault));
            unit.store(omegas);
            auto last = unit;
            for (u32 i = 1; i < 32; i++) {
                last = last.square();
                last.store(omegas + i * WORDS);
            }

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

        ~recompute_ntt() override {
            cudaFreeHost(pq);
            cudaFreeHost(omegas);
            if (inverse) cudaFreeHost(inv_n);
            if (process) cudaFreeHost(zeta);
            if (on_gpu) clean_gpu();
        }

        cudaError_t to_gpu(cudaStream_t stream = 0) override {
            std::unique_lock<std::shared_mutex> wlock(this->mtx);
            cudaError_t first_err = cudaSuccess;

            CUDA_CHECK(cudaMallocAsync(&pq_d, (1 << max_deg >> 1) * sizeof(u32_E) * WORDS, stream))
            CUDA_CHECK(cudaMallocAsync(&omegas_d, 32 * sizeof(u32_E) * WORDS, stream))

            CUDA_CHECK(cudaMemcpyAsync(pq_d, pq, (1 << max_deg >> 1) * sizeof(u32_E) * WORDS, cudaMemcpyHostToDevice, stream))
            CUDA_CHECK(cudaMemcpyAsync(omegas_d, omegas, 32 * sizeof(u32_E) * WORDS, cudaMemcpyHostToDevice, stream))

            if (inverse) {
                CUDA_CHECK(cudaMallocAsync(&inv_n_d, WORDS * sizeof(u32), stream));
                CUDA_CHECK(cudaMemcpyAsync(inv_n_d, inv_n, WORDS * sizeof(u32), cudaMemcpyHostToDevice, stream));
            }

            if (process) {
                CUDA_CHECK(cudaMallocAsync(&zeta_d, 2 * WORDS * sizeof(u32), stream));
                CUDA_CHECK(cudaMemcpyAsync(zeta_d, zeta, 2 * WORDS * sizeof(u32), cudaMemcpyHostToDevice, stream));
            }

            if (first_err != cudaSuccess) {
                CUDA_CHECK(cudaFreeAsync(pq_d, stream));
                CUDA_CHECK(cudaFreeAsync(omegas_d, stream));
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

            CUDA_CHECK(cudaFreeAsync(pq_d, stream));
            CUDA_CHECK(cudaFreeAsync(omegas_d, stream));

            if (inverse) CUDA_CHECK(cudaFreeAsync(inv_n_d, stream));
            if (process) CUDA_CHECK(cudaFreeAsync(zeta_d, stream));

            this->on_gpu = false;
            return first_err;
        }

        cudaError_t ntt(u32 * data, cudaStream_t stream = 0, u32 start_n = 0, u32 **dev_ptr = nullptr) override {
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
            CUDA_CHECK(cudaMallocAsync(&x, len * WORDS * sizeof(u32), stream));
            if (process && !inverse) {
                CUDA_CHECK(cudaMemcpyAsync(x, data, start_n * WORDS * sizeof(u32), cudaMemcpyHostToDevice, stream));
            } else {
                CUDA_CHECK(cudaMemcpyAsync(x, data, len * WORDS * sizeof(u32), cudaMemcpyHostToDevice, stream));
            }
            if (dev_ptr != nullptr) *dev_ptr = x;

            if (debug) {
                dim3 block(768);
                dim3 grid((len - 1) / block.x + 1);
                number_to_element <Field> <<< grid, block, 0, stream >>> (x, len);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventRecord(start));
            }

            int log_stride = log_len - 1;
            constexpr u32 io_group = 1 << (log2_int(WORDS - 1) + 1);
            
            while (log_stride >= log_len / 2) {
                u32 deg = std::min((int)max_deg_stage1, (log_stride + 1 - log_len / 2));

                u32 group_num = std::min((int)(len / (1 << deg)), 1 << (max_threads_stage1_log - (deg - 1)));

                u32 block_sz = (1 << (deg - 1)) * group_num;
                assert(block_sz <= (1 << max_threads_stage1_log));
                u32 block_num = len / 2 / block_sz;
                assert(block_num * 2 * block_sz == len);

                dim3 block(block_sz);
                dim3 grid(block_num);

                auto kernel = SSIP_NTT_stage1_warp_recompute <Field, io_group>;

                u32 shared_size = (sizeof(u32) * ((1 << deg) + 1) * WORDS) * group_num;

                kernel <<< grid, block, shared_size, stream >>>(x, pq_d, log_len, log_stride, deg, max_deg, 1 << (deg - 1), omegas_d);
                CUDA_CHECK(cudaGetLastError());                

                log_stride -= deg;
            }

            assert (log_stride == log_len / 2 - 1);

            while (log_stride >= 0) {
                u32 deg = std::min((int)max_deg_stage2, log_stride + 1);

                u32 group_num = std::min((int)(len / (1 << (deg << 1))), 1 << (max_threads_stage2_log - 2 * (deg - 1)));

                u32 block_sz = (1 << ((deg - 1) << 1)) * group_num;
                assert(block_sz <= (1 << max_threads_stage2_log));
                u32 block_num = len / 4 / block_sz;
                assert(block_num * 4 * block_sz == len);

                dim3 block(block_sz);
                dim3 grid(block_num);

                u32 shared_size = (sizeof(u32) * ((1 << (deg << 1)) + 1) * WORDS) * group_num;

                auto kernel = SSIP_NTT_stage2_warp_recompute <Field, io_group>;

                kernel <<< grid, block, shared_size, stream >>>(x, pq_d, log_len, log_stride, deg, max_deg, ((1 << (deg << 1)) >> 2), omegas_d);
                CUDA_CHECK(cudaGetLastError());   

                log_stride -= deg;
            }

            if (debug) {
                CUDA_CHECK(cudaEventRecord(end));
                CUDA_CHECK(cudaEventSynchronize(end));
                CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
                dim3 block(768);
                dim3 grid((len - 1) / block.x + 1);
                element_to_number <Field> <<< grid, block, 0, stream >>> (x, len);
                CUDA_CHECK(cudaGetLastError());
            }

            rlock.unlock();

            if ((first_err == cudaSuccess) && dev_ptr == nullptr) CUDA_CHECK(cudaMemcpyAsync(data, x, len * WORDS * sizeof(u32), cudaMemcpyDeviceToHost, stream));

            if (dev_ptr == nullptr)CUDA_CHECK(cudaFreeAsync(x, stream));

            if (dev_ptr == nullptr) CUDA_CHECK(cudaStreamSynchronize(stream));

            // manually tackle log_len == 1 case because the stage2 kernel won't run
            if (log_len == 1) {
                if (inverse) {
                    (Field::load(data) * Field::load(inv_n)).store(data);
                    (Field::load(data + WORDS) * Field::load(inv_n)).store(data + WORDS);
                    if (process) {
                        (Field::load(data + WORDS) * Field::load(zeta)).store(data + WORDS);
                    }
                }
            }
            this->sem.release();

            if (debug) CUDA_CHECK(clean_gpu(stream));

            return first_err;
        }
    };
}