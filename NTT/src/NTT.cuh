#pragma once

#include "../../mont/src/mont.cuh"
#include <cuda_runtime.h>
#include <cassert>

namespace NTT {
    typedef uint u32;
    typedef unsigned long long u64;
    typedef uint u32_E;
    typedef uint u32_N;

    __constant__ u32_E omegas_c[32 * 8]; // need to be revised if words changed

    template<u32 WORDS>
    __global__ void element_to_number(u32* data, u32 len, mont256::Params* param) {
        auto env = mont256::Env(*param);
        u64 index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= len) return;
        env.to_number(mont256::Element::load(data + index * WORDS)).store(data + index * WORDS);
    }

    template<u32 WORDS>
    __global__ void number_to_element(u32* data, u32 len, mont256::Params* param) {
        auto env = mont256::Env(*param);
        u64 index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= len) return;
        env.from_number(mont256::Number::load(data + index * WORDS)).store(data + index * WORDS);
    }
    
    template<u32 WORDS>
    __global__ void rearrange(u32_E * data, uint2 * reverse, u32 len) {
        u32 index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= len) return;
        uint2 r = reverse[index];
        
        #pragma unroll
        for (u32 i = 0; i < WORDS; i++) {
            u32 tmp = data[((u64)r.x) * WORDS + i];
            data[((u64)r.x) * WORDS + i] = data[((u64)r.y) * WORDS + i];
            data[((u64)r.y) * WORDS + i] = tmp;
        }
    }

    template<u32 WORDS>
    __global__ void naive(u32_E* data, u32 len, u32_E* roots, u32 stride, mont256::Params* param) {
        auto env = mont256::Env(*param);

        u32 id = (blockDim.x * blockIdx.x + threadIdx.x);
        if (id << 1 >= len) return;

        u64 offset = id % stride;
        u64 pos = ((id - offset) << 1) + offset;

        auto a = mont256::Element::load(data + (pos * WORDS));
        auto b = mont256::Element::load(data + ((pos + stride) * WORDS));
        auto w = mont256::Element::load(roots + (offset * len / (stride << 1)) * WORDS);

        b = env.mul(b, w);
        
        auto tmp = a;
        a = env.add(a, b);
        b = env.sub(tmp, b);
        
        a.store(data + (pos * WORDS));
        b.store(data + ((pos + stride) * WORDS));
    }


    template<u32 WORDS>
    class naive_ntt {
        const mont256::Params param;
        mont256::Element unit;
        bool debug;
        uint2 * reverse;
        u32_E * roots;
        const u32 log_len, len;
        u32 r_len;
        
        u32 gen_reverse(u32 log_len, uint2* reverse_pair) {
            u32 len = 1 << log_len;
            u32* reverse = new u32[len];
            for (u32 i = 0; i < len; i++) {
                reverse[i] = (reverse[i >> 1] >> 1) | ((i & 1) << (log_len - 1) ); //reverse the bits
            }
            int r_len = 0;
            for (u32 i = 0; i < len; i++) {
                if (reverse[i] < i) {
                    reverse_pair[r_len].x = i;
                    reverse_pair[r_len].y = reverse[i];
                    r_len++;
                }
            }
            delete[] reverse;
            return r_len;
        }

        void gen_roots(u32_E * roots, u32 len) {
            auto env = mont256::Env::host_new(param);
            env.one().store(roots);
            auto last_element = mont256::Element::load(roots);
            for (u64 i = WORDS; i < ((u64)len) * WORDS; i+= WORDS) {
                last_element = env.host_mul(last_element, unit);
                last_element.store(roots + i);
            }
        }

        public:
        float milliseconds = 0;

        naive_ntt(const mont256::Params &param, const u32_N* omega, u32 log_len, bool debug) : param(param), log_len(log_len), len(1 << log_len), debug(debug) {
            
            auto env = mont256::Env::host_new(param);
            // unit = qpow(omega, (P - 1ll) / len)
            unit = env.host_from_number(mont256::Number::load(omega));
            auto exponent = mont256::Number::load(param.m);
            auto one = mont256::Number::zero();
            one.c0 = 1;
            exponent = exponent.host_sub(one);
            exponent = exponent.slr(log_len);
            unit = env.host_pow(unit, exponent);

            roots = (u32_E *) malloc(((u64)len) * WORDS * sizeof(u32_E));
            gen_roots(roots, len);

            reverse = (uint2 *) malloc(((u64)len) * sizeof(uint2));
            r_len = gen_reverse(log_len, reverse);
        }

        ~naive_ntt() {
            free(roots);
            free(reverse);
        }

        void ntt(u32 * data) {
            cudaEvent_t start, end;
            cudaEventCreate(&start);
            cudaEventCreate(&end);

            uint2 * reverse_d;
            cudaMalloc(&reverse_d, r_len * sizeof(uint2));
            cudaMemcpy(reverse_d, reverse, r_len * sizeof(uint2), cudaMemcpyHostToDevice);
            
            u32_E * data_d;
            cudaMalloc(&data_d, ((u64)len) * WORDS * sizeof(u32_E));
            cudaMemcpy(data_d, data, ((u64)len) * WORDS * sizeof(u32_E), cudaMemcpyHostToDevice);

            u32_E * roots_d;
            cudaMalloc(&roots_d, ((u64)len) * WORDS * sizeof(u32_E));
            cudaMemcpy(roots_d, roots, ((u64)len) * WORDS * sizeof(u32_E), cudaMemcpyHostToDevice);

            mont256::Params *param_d;
            cudaMalloc(&param_d, sizeof(mont256::Params));
            cudaMemcpy(param_d, &param, sizeof(mont256::Params), cudaMemcpyHostToDevice);

            dim3 rearrange_block(768);
            dim3 rearrange_grid((r_len + rearrange_block.x - 1) / rearrange_block.x);
            dim3 ntt_block(768);
            dim3 ntt_grid(((len >> 1) - 1) / ntt_block.x + 1);

            if (debug) {
                dim3 block(768);
                dim3 grid((len - 1) / block.x + 1);
                number_to_element <WORDS> <<< grid, block >>> (data_d, len, param_d);
                cudaEventRecord(start);
            }
            
            rearrange <WORDS> <<<rearrange_grid, rearrange_block>>> (data_d, reverse_d, r_len);
            for (u32 stride = 1; stride < len; stride <<= 1) {
                naive <WORDS> <<<ntt_grid, ntt_block>>> (data_d, len, roots_d, stride, param_d);
            }

            if (debug) {
                cudaEventRecord(end);
                cudaEventSynchronize(end);
                cudaEventElapsedTime(&milliseconds, start, end);
                dim3 block(768);
                dim3 grid((len - 1) / block.x + 1);
                element_to_number <WORDS> <<< grid, block >>> (data_d, len, param_d);
            }

            cudaMemcpy(data, data_d, ((u64)len) * WORDS * sizeof(u32), cudaMemcpyDeviceToHost);

            cudaFree(data_d);
            cudaFree(reverse_d);
            cudaFree(roots_d); 
        }
    };

    template <u32 WORDS>
    __forceinline__ __device__ mont256::Element pow_lookup(u32 exponent, mont256::Env &env) {
        auto res = env.one();
        u32 i = 0;
        while(exponent > 0) {
            if (exponent & 1) {
                res = env.mul(res, mont256::Element::load(omegas_c + (i * WORDS)));
            }
            exponent = exponent >> 1;
            i++;
        }
        return res;
    }

    /*
    * FFT algorithm is inspired from: http://www.bealto.com/gpu-fft_group-1.html
    */
    template <u32 WORDS>
    __global__ void bellperson_kernel(u32_E * x, // Source buffer
                        u32_E * y, // Destination buffer
                        const u32_E * pq, // Precalculated twiddle factors
                        u32 n, // Number of elements
                        u32 lgp, // Log2 of `p` (Read more in the link above)
                        u32 deg, // 1=>radix2, 2=>radix4, 3=>radix8, ...
                        u32 max_deg, // Maximum degree supported, according to `pq` and `omegas`
                        mont256::Params* param)
    {

        // There can only be a single dynamic shared memory item, hence cast it to the type we need.
        extern __shared__ u32 u[];

        auto env = mont256::Env(*param);

        u32 lid = threadIdx.x;//GET_LOCAL_ID();
        u32 lsize = blockDim.x;//GET_LOCAL_SIZE();
        u32 index = blockIdx.x;//GET_GROUP_ID();
        u32 t = n >> deg;
        u32 p = 1 << lgp;
        u32 k = index & (p - 1);

        x += index * WORDS;
        y += (((index - k) << deg) + k) * WORDS;

        u32 count = 1 << deg; // 2^deg
        u32 counth = count >> 1; // Half of count

        u32 counts = count / lsize * lid;
        u32 counte = counts + count / lsize;

        // Compute powers of twiddle
        auto twiddle = pow_lookup<WORDS>((n >> lgp >> deg) * k, env);

        auto tmp = env.pow(twiddle, counts, deg);

        for(u64 i = counts; i < counte; i++) {
            auto num = mont256::Element::load(x + (i * t * WORDS));
            num = env.mul(num, tmp);
            num.store(u + (i * WORDS));
            tmp = env.mul(tmp, twiddle);
        }

        __syncthreads();

        const u32 pqshift = max_deg - deg;
        for(u32 rnd = 0; rnd < deg; rnd++) {
            const u32 bit = counth >> rnd;
            for(u32 i = counts >> 1; i < counte >> 1; i++) {
                const u32 di = i & (bit - 1);
                const u32 i0 = (i << 1) - di;
                const u32 i1 = i0 + bit;
                mont256::Element a, b, tmp, w;
                a = mont256::Element::load(u + (i0 * WORDS));
                b = mont256::Element::load(u + (i1 * WORDS));
                tmp = a;
                a = env.add(a, b);
                b = env.sub(tmp, b);
                if (di != 0) {
                    w = mont256::Element::load(pq + ((di << rnd << pqshift) * WORDS));
                    b = env.mul(b, w);
                }
                a.store(u + (i0 * WORDS));
                b.store(u + (i1 * WORDS));
            }
            __syncthreads();
        }
        

        for(u32 i = counts >> 1; i < counte >> 1; i++) {
            #pragma unroll
            for (u32 j = 0; j < WORDS; j++) {
                y[(((u64)i) * p) * WORDS + j] = u[(__brev(i) >> (32 - deg)) * WORDS + j];
                y[(((u64)(i + counth)) * p) * WORDS + j] = u[(__brev(i + counth) >> (32 - deg)) * WORDS + j];
            }
        }
    }
    
    template<u32 WORDS>
    class bellperson_ntt {
        const u32 max_deg;
        const u32 log_len;
        const u64 len;
        mont256::Params param;
        mont256::Element unit;
        bool debug;
        u32_E *pq; // Precalculated values for radix degrees up to `max_deg`
        u32_E *omegas; // Precalculated values for [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]


        public:
        float milliseconds = 0;
        bellperson_ntt(mont256::Params param, u32* omega, u32 log_len, bool debug) : param(param), log_len(log_len), max_deg(std::min(8u, log_len)), len(1 << log_len), debug(debug) {
            // Precalculate:
            auto env = mont256::Env::host_new(param);

            // unit = qpow(omega, (P - 1ll) / len)
            unit = env.host_from_number(mont256::Number::load(omega));
            auto exponent = mont256::Number::load(param.m);
            auto one = mont256::Number::zero();
            one.c0 = 1;
            exponent = exponent.host_sub(one);
            exponent = exponent.slr(log_len);
            unit = env.host_pow(unit, exponent);

            // pq: [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]

            pq = (u32 *) malloc((1 << max_deg >> 1) * sizeof(u32) * WORDS);
            memset(pq, 0, (1 << max_deg >> 1) * sizeof(u32) * WORDS);
            env.one().store(pq);
            auto twiddle = env.host_pow(unit, len >> max_deg);
            if (max_deg > 1) {
                twiddle.store(pq + WORDS);
                auto last = twiddle;
                for (u32 i = 2; i < (1 << max_deg >> 1); i++) {
                    last = env.host_mul(last, twiddle);
                    last.store(pq + i * WORDS);
                }
            }

            // omegas: [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]

            omegas = (u32 *) malloc(32 * sizeof(u32) * WORDS);
            unit.store(omegas);
            auto last = unit;
            for (u32 i = 1; i < 32; i++) {
                last = env.host_square(last);
                last.store(omegas + i * WORDS);
            }
        }

        ~bellperson_ntt() {
            free(pq);
            free(omegas);
        }

        void ntt(u32 * data) {
            cudaEvent_t start, end;
            cudaEventCreate(&start);
            cudaEventCreate(&end);
         
            u32_E* pq_d;

            cudaMalloc(&pq_d, (1 << max_deg >> 1) * sizeof(u32_E) * WORDS);
            cudaMemcpy(pq_d, pq, (1 << max_deg >> 1) * sizeof(u32_E) * WORDS, cudaMemcpyHostToDevice);

            cudaMemcpyToSymbol(omegas_c, omegas, 32 * sizeof(u32_E) * WORDS);

            u32 * x, * y;
            cudaMalloc(&y, len * WORDS * sizeof(u32));
            cudaMalloc(&x, len * WORDS * sizeof(u32));
            cudaMemcpy(x, data, len * WORDS * sizeof(u32), cudaMemcpyHostToDevice);

            mont256::Params* param_d;
            cudaMalloc(&param_d, sizeof(mont256::Params));
            cudaMemcpy(param_d, &param, sizeof(mont256::Params), cudaMemcpyHostToDevice);

            // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
            u32 log_p = 0u;
            
            if (debug) {
                dim3 block(768);
                dim3 grid((len - 1) / block.x + 1);
                number_to_element <WORDS> <<< grid, block >>> (x, len, param_d);
                cudaEventRecord(start);
            }

            // Each iteration performs a FFT round
            while (log_p < log_len) {

                // 1=>radix2, 2=>radix4, 3=>radix8, ...
                u32 deg = std::min(max_deg, log_len - log_p);

                assert(deg - 1 <= 10);

                dim3 block(1 << (deg - 1));
                dim3 grid(len >> deg);

                uint shared_size = (1 << deg) * WORDS * sizeof(u32);

                auto kernel = bellperson_kernel<WORDS>;
            
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size);

                kernel <<< grid, block, shared_size >>> (x, y, pq_d, len, log_p, deg, max_deg, param_d);
                
                auto err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("Error: %s\n", cudaGetErrorString(err));
                }
                log_p += deg;

                u32 * tmp = x;
                x = y;
                y = tmp;
            }

            if (debug) {
                cudaEventRecord(end);
                cudaEventSynchronize(end);
                cudaEventElapsedTime(&milliseconds, start, end);
                dim3 block(768);
                dim3 grid((len - 1) / block.x + 1);
                element_to_number <WORDS> <<< grid, block >>> (x, len, param_d);
            }

            cudaMemcpy(data, x, len * WORDS * sizeof(u32), cudaMemcpyDeviceToHost);

            cudaFree(pq_d);
            cudaFree(y);
            cudaFree(x);

        }
    };

    template <u32 WORDS>
    __global__ void SSIP_NTT_stage1 (u32_E * x, const u32_E * pq, u32 len, u32 log_stride, u32 deg, u32 max_deg, u32 io_group, mont256::Params* param, u32 group_sz) {
        extern __shared__ u32_E s[];
        // column-major in shared memory
        // data patteren:
        // a0_word0 a1_word0 a2_word0 a3_word0 ... an_word0 [empty] a0_word1 a1_word1 a2_word1 a3_word1 ...
        // for coleasced read, we need to read a0_word0, a0_word1, a0_word2, ... a0_wordn in a single read
        // so thread [0,WORDS) will read a0_word0 a0_word1 a0_word2 ... 
        // then a1_word0 a1_word1 a1_word2 ... 
        // so we need the empty space is for padding to avoid bank conflict during read because n is likely to be 32k

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
        const u32 subblock_id = segment_id & (end_stride - 1);

        x += ((u64)(segment_start + subblock_id)) * WORDS; // use u64 to avoid overflow

        const u32 io_id = lid & (io_group - 1);
        const u32 lid_start = lid - io_id;
        const u32 shared_read_stride = (lsize << 1) + 1;
        const u32 cur_io_group = io_group < lsize ? io_group : lsize;
        const u32 io_per_thread = io_group / cur_io_group;

        // Read data
        for (u32 i = lid_start; i < lid_start + cur_io_group; i++) {
            for (u32 j = 0; j < io_per_thread; j++) {
                u32 io = io_id + j * cur_io_group;
                if (io < WORDS) {
                    u32 group_id = i & (subblock_sz - 1);
                    u32 gpos = group_id << (lgp + 1);
                    u[(i << 1) + io * shared_read_stride] = x[gpos * WORDS + io];
                    u[(i << 1) + 1 + io * shared_read_stride] = x[(gpos + end_stride) * WORDS + io];
                }
            }
        }

        auto env = mont256::Env(*param);

        __syncthreads();

        const u32 pqshift = max_deg - deg;
        for(u32 rnd = 0; rnd < deg; rnd++) {
            const u32 bit = subblock_sz >> rnd;
            const u32 di = lid & (bit - 1);
            const u32 i0 = (lid << 1) - di;
            const u32 i1 = i0 + bit;

            auto a = mont256::Element::load(u + i0, shared_read_stride);
            auto b = mont256::Element::load(u + i1, shared_read_stride);
            auto tmp = a;
            a = env.add(a, b);
            b = env.sub(tmp, b);
            if (di != 0) {
                auto w = mont256::Element::load(pq + (di << rnd << pqshift) * WORDS);
                b = env.mul(b, w);
            }
            a.store(u + i0, shared_read_stride);
            b.store(u + i1, shared_read_stride);

            __syncthreads();
        }

        // Twiddle factor
        u32 k = index & (end_stride - 1);
        auto twiddle = pow_lookup <WORDS> ((len >> (log_stride - deg + 1) >> deg) * k, env);

        auto t1 = env.pow(twiddle, lid << 1, deg);

        auto pos1 = __brev(lid << 1) >> (32 - deg);
        auto pos2 = __brev((lid << 1) + 1) >> (32 - deg);

        auto a = mont256::Element::load(u + pos1, shared_read_stride);
        a = env.mul(a, t1);
        a.store(u + pos1, shared_read_stride);
        
        auto t2 = env.mul(t1, twiddle);

        auto b = mont256::Element::load(u + pos2, shared_read_stride);
        b = env.mul(b, t2);
        b.store(u + pos2, shared_read_stride);

        __syncthreads();

        // Write back
        for (u32 i = lid_start; i < lid_start + cur_io_group; i++) {
            for (u32 j = 0; j < io_per_thread; j++) {
                u32 io = io_id + j * cur_io_group;
                if (io < WORDS) {
                    u32 group_id = i & (subblock_sz - 1);
                    u32 gpos = group_id << (lgp + 1);
                    x[gpos * WORDS + io] = u[(i << 1) + io * shared_read_stride];
                    x[(gpos + end_stride) * WORDS + io] = u[(i << 1) + 1 + io * shared_read_stride];
                }
            }
        }
    }

    template <u32 WORDS>
    __global__ void SSIP_NTT_stage2 (u32_E * data, const u32_E * pq, u32 log_len, u32 log_stride, u32 deg, u32 max_deg, u32 io_group, mont256::Params* param, u32 group_sz) {
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

        // Read data
        for (u32 i = lid_start; i < lid_start + cur_io_group; i++) {
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
        
        auto env = mont256::Env(*param);

        __syncthreads();

        const u32 pqshift = max_deg - deg;
        for(u32 rnd = 0; rnd < deg; rnd++) {
        
            const u32 bit = subblock_sz >> rnd;
            const u32 gap = (lsize << 1) >> (deg - rnd - 1);
            const u32 offset = (gap) * (lid / (gap >> 1));

            const u32 di = lid & (bit - 1);
            const u32 i0 = (lid << 1) - di + offset;
            const u32 i1 = i0 + bit;
            const u32 i2 = i0 + gap;
            const u32 i3 = i0 + gap + bit;

            auto a = mont256::Element::load(u + i0, shared_read_stride);
            auto b = mont256::Element::load(u + i1, shared_read_stride);
            auto c = mont256::Element::load(u + i2, shared_read_stride);
            auto d = mont256::Element::load(u + i3, shared_read_stride);
            auto tmp1 = a;
            auto tmp2 = c;

            a = env.add(a, b);
            c = env.add(c, d);
            b = env.sub(tmp1, b);
            d = env.sub(tmp2, d);

            if (di != 0) {
                auto w = mont256::Element::load(pq + (di << rnd << pqshift) * WORDS);
                b = env.mul(b, w);
                d = env.mul(d, w);
            }

            a.store(u + i0, shared_read_stride);
            b.store(u + i1, shared_read_stride);
            c.store(u + i2, shared_read_stride);
            d.store(u + i3, shared_read_stride);

            __syncthreads();
        }

        // Twiddle factor
        u32 k = index & (end_stride - 1);
        u32 n = 1 << log_len;
        auto twiddle = pow_lookup<WORDS>((n >> (log_stride - deg + 1) >> deg) * k, env);
        auto twiddle_gap = pow_lookup<WORDS>((n >> (log_stride - deg + 1) >> deg) * k * (1 << (deg - 1)), env);
        
        u32 a, b, c, d;
        a = __brev(lid << 1) >> (32 - (deg << 1));
        b = __brev((lid << 1) + 1) >> (32 - (deg << 1));
        c = __brev((lid << 1) + (lsize << 1)) >> (32 - (deg << 1));
        d = __brev((lid << 1) + (lsize << 1) + 1) >> (32 - (deg << 1));

        auto t1 = env.pow(twiddle, lid << 1 >> deg, deg);

        auto num = mont256::Element::load(u + a, shared_read_stride);
        num = env.mul(num, t1);
        num.store(u + a, shared_read_stride);

        num = mont256::Element::load(u + b, shared_read_stride);
        num = env.mul(num, t1);
        num.store(u + b, shared_read_stride);

        auto t2 = env.mul(t1, twiddle_gap);//env.pow(twiddle, ((lid << 1) >> deg) + ((lsize <<1) >> deg), deg);

        num = mont256::Element::load(u + c, shared_read_stride);
        num = env.mul(num, t2);
        num.store(u + c, shared_read_stride);

        num = mont256::Element::load(u + d, shared_read_stride);
        num = env.mul(num, t2);
        num.store(u + d, shared_read_stride);

        __syncthreads();

        // Write back
        for (u32 i = lid_start; i < lid_start + cur_io_group; i++) {
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

    template <u32 WORDS>
    class self_sort_in_place_ntt {
        const u32 max_threads_stage1_log = 9;
        const u32 max_threads_stage2_log = 8;
        u32 max_deg_stage1;
        u32 max_deg_stage2;
        u32 max_deg;
        const u32 log_len;
        const u32 io_group = 1 << ((int)log2(WORDS - 1) + 1);
        const u64 len;
        mont256::Params param;
        mont256::Element unit;
        bool debug;
        u32_E *pq; // Precalculated values for radix degrees up to `max_deg`
        u32_E *omegas; // Precalculated values for [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]

        u32 get_deg (u32 deg_stage, u32 max_deg_stage) {
            u32 deg_per_round;
            for (u32 rounds = 1; ; rounds++) {
                deg_per_round = rounds == 1 ? deg_stage : (deg_stage + 1) / rounds;
                if (deg_per_round <= max_deg_stage) break;
            }
            return deg_per_round;
        }

        public:
        float milliseconds = 0;
        self_sort_in_place_ntt(mont256::Params param, u32* omega, u32 log_len, bool debug) 
        : param(param), log_len(log_len), len(1 << log_len), debug(debug) {
            u32 deg_stage1 = (log_len + 1) / 2;
            u32 deg_stage2 = log_len / 2;
            max_deg_stage1 = get_deg(deg_stage1, max_threads_stage1_log + 1);
            max_deg_stage2 = get_deg(deg_stage2, (max_threads_stage2_log + 2) / 2);
            max_deg = std::max(max_deg_stage1, max_deg_stage2);

            // Precalculate:
            auto env = mont256::Env::host_new(param);

            // unit = qpow(omega, (P - 1ll) / len)
            unit = env.host_from_number(mont256::Number::load(omega));
            auto exponent = mont256::Number::load(param.m);
            auto one = mont256::Number::zero();
            one.c0 = 1;
            exponent = exponent.host_sub(one);
            exponent = exponent.slr(log_len);
            unit = env.host_pow(unit, exponent);

            // pq: [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]

            pq = (u32 *) malloc((1 << max_deg >> 1) * sizeof(u32) * WORDS);
            memset(pq, 0, (1 << max_deg >> 1) * sizeof(u32) * WORDS);
            env.one().store(pq);
            auto twiddle = env.host_pow(unit, len >> max_deg);
            if (max_deg > 1) {
                twiddle.store(pq + WORDS);
                auto last = twiddle;
                for (u32 i = 2; i < (1 << max_deg >> 1); i++) {
                    last = env.host_mul(last, twiddle);
                    last.store(pq + i * WORDS);
                }
            }

            // omegas: [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]

            omegas = (u32 *) malloc(32 * sizeof(u32) * WORDS);
            unit.store(omegas);
            auto last = unit;
            for (u32 i = 1; i < 32; i++) {
                last = env.host_square(last);
                last.store(omegas + i * WORDS);
            }
        }

        ~self_sort_in_place_ntt() {
            free(pq);
            free(omegas);
        }

        void ntt(u32 * data) {
            cudaEvent_t start, end;
            cudaEventCreate(&start);
            cudaEventCreate(&end);
         
            u32_E* pq_d;

            cudaMalloc(&pq_d, (1 << max_deg >> 1) * sizeof(u32_E) * WORDS);
            cudaMemcpy(pq_d, pq, (1 << max_deg >> 1) * sizeof(u32_E) * WORDS, cudaMemcpyHostToDevice);

            cudaMemcpyToSymbol(omegas_c, omegas, 32 * sizeof(u32_E) * WORDS);

            u32 * x;
            cudaMalloc(&x, len * WORDS * sizeof(u32));
            cudaMemcpy(x, data, len * WORDS * sizeof(u32), cudaMemcpyHostToDevice);

            mont256::Params* param_d;
            cudaMalloc(&param_d, sizeof(mont256::Params));
            cudaMemcpy(param_d, &param, sizeof(mont256::Params), cudaMemcpyHostToDevice);
            

            if (debug) {
                dim3 block(768);
                dim3 grid((len - 1) / block.x + 1);
                number_to_element <WORDS> <<< grid, block >>> (x, len, param_d);
                cudaEventRecord(start);
            }

            int log_stride = log_len - 1;
            
            while (log_stride >= log_len / 2) {
                u32 deg = std::min(max_deg_stage1, (log_stride + 1 - log_len / 2));

                u32 group_num = std::min((int)(len / (1 << deg)), 1 << (max_threads_stage1_log - (deg - 1)));

                u32 block_sz = (1 << (deg - 1)) * group_num;
                assert(block_sz <= (1 << max_threads_stage1_log));
                u32 block_num = len / 2 / block_sz;
                assert(block_num * 2 * block_sz == len);

                dim3 block(block_sz);
                dim3 grid(block_num);

                u32 shared_size = (sizeof(u32) * ((1 << deg) + 1) * WORDS) * group_num;

                auto kernel = SSIP_NTT_stage1 <WORDS>;
            
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size);

                kernel <<< grid, block, shared_size >>>(x, pq_d, len, log_stride, deg, max_deg, io_group, param_d, 1 << (deg - 1));

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

                dim3 block1(block_sz);
                dim3 grid1(block_num);

                uint shared_size = (sizeof(u32) * ((1 << (deg << 1)) + 1) * WORDS) * group_num;

                auto kernel = SSIP_NTT_stage2 <WORDS>;
            
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size);

                kernel <<< grid1, block1, shared_size >>>(x, pq_d, log_len, log_stride, deg, max_deg, io_group, param_d, ((1 << (deg << 1)) >> 2));

                log_stride -= deg;
            }

            if (debug) {
                cudaEventRecord(end);
                cudaEventSynchronize(end);
                cudaEventElapsedTime(&milliseconds, start, end);
                dim3 block(768);
                dim3 grid((len - 1) / block.x + 1);
                element_to_number <WORDS> <<< grid, block >>> (x, len, param_d);
            }

            cudaMemcpy(data, x, len * WORDS * sizeof(u32), cudaMemcpyDeviceToHost);

            cudaFree(pq_d);
            cudaFree(x);

        }
    };
}