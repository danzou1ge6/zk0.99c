#pragma once

#include "../../mont/src/mont.cuh"
#include <cuda_runtime.h>

namespace NTT {
    typedef uint u32;
    typedef unsigned long long u64;
    typedef uint u32_E;
    typedef uint u32_N;

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
                number_to_element <WORDS> <<< block, grid >>> (data_d, len, param_d);
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
                element_to_number <WORDS> <<< block, grid >>> (data_d, len, param_d);
            }

            cudaMemcpy(data, data_d, ((u64)len) * WORDS * sizeof(u32), cudaMemcpyDeviceToHost);

            cudaFree(data_d);
            cudaFree(reverse_d);
            cudaFree(roots_d); 
        }
    };

    template <u32 WORDS>
    __forceinline__ __device__ mont256::Element pow_lookup(u32 *omegas, u32 exponent, mont256::Env &env) {
        auto res = env.one();
        u32 i = 0;
        while(exponent > 0) {
            if (exponent & 1) {
                res = env.mul(res, mont256::Element::load(omegas + i * WORDS));
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
    __global__ void bellperson_kernel(u32 * x, // Source buffer
                        u32 * y, // Destination buffer
                        u32 * pq, // Precalculated twiddle factors
                        u32 * omegas, // [omega, omega^2, omega^4, ...]
                        u32 n, // Number of elements
                        u32 lgp, // Log2 of `p` (Read more in the link above)
                        u32 deg, // 1=>radix2, 2=>radix4, 3=>radix8, ...
                        u32 max_deg, // Maximum degree supported, according to `pq` and `omegas`
                        mont256::Params* param)
    {

        // There can only be a single dynamic shared memory item, hence cast it to the type we need.
        extern __shared__ u32* u;

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
        auto twiddle = pow_lookup<WORDS>(omegas, (n >> lgp >> deg) * k, env);

        auto tmp = env.pow(twiddle, counts);

        for(u32 i = counts; i < counte; i++) {
            auto num = mont256::Element::load(x + i * t * WORDS);
            num = env.mul(num, tmp);
            num.store(u + i * WORDS);
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
                a = a.load(u + i0 * WORDS);
                b = b.load(u + i1 * WORDS);
                tmp = a;
                a = env.add(a, b);
                b = env.sub(tmp, b);
                if (di != 0) {
                    w = w.load(pq + (di << rnd << pqshift) * WORDS);
                    b = env.mul(b, w);
                }
                a.store(u + i0 * WORDS);
                b.store(u + i1 * WORDS);
            }
            __syncthreads();
        }
        

        for(u32 i = counts >> 1; i < counte >> 1; i++) {
            #pragma unroll
            for (u32 j = 0; j < WORDS; j++) {
                y[(i * p) * WORDS + j] = u[(__brev(i) >> (32 - deg)) * WORDS + j];
                y[((i + counth) * p) * WORDS + j] = u[(__brev(i + counth) >> (32 - deg)) * WORDS + j];
            }
        }
    }
    
    template<u32 WORDS>
    class bellperson_ntt {
        const u32 max_deg = 8u;
        const u32 log_len, len;
        mont256::Params param;
        mont256::Element unit;
        bool timer;
        u32 *pq; // Precalculated values for radix degrees up to `max_deg`
        u32 *omegas; // Precalculated values for [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]


        public:
        float milliseconds = 0;
        bellperson_ntt(mont256::Params param, u32* omega, u32 log_len, bool timer) : param(param), log_len(log_len), len(1 << log_len), timer(timer) {
            // Precalculate:
            auto env = mont256::Env(param);

            // unit = qpow(omega, (P - 1ll) / len)
            unit = unit.load(omega);
            auto exponent = mont256::Number::load(param.m);
            auto one = mont256::Number::zero();
            one.c0 = 1;
            exponent = exponent.host_sub(one) ;
            exponent.slr(log_len);            
            unit = env.host_pow(unit, exponent);

            // pq: [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]

            pq = (u32 *) malloc((1 << max_deg >> 1) * sizeof(u32) * WORDS);
            memset(pq, 0, (1 << max_deg >> 1) * sizeof(u32) * WORDS);
            pq[0] = 1;
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
         
            u32* pq_d;
            u32* omegas_d;

            cudaMalloc(&pq_d, (1 << max_deg >> 1) * sizeof(u32) * WORDS);
            cudaMemcpy(pq_d, pq, (1 << max_deg >> 1) * sizeof(u32) * WORDS, cudaMemcpyHostToDevice);

            cudaMalloc(&omegas_d, 32 * sizeof(u32) * WORDS);
            cudaMemcpy(omegas_d, omegas, 32 * sizeof(u32) * WORDS, cudaMemcpyHostToDevice);

            u32 * x, * y;
            cudaMalloc(&y, len * WORDS * sizeof(u32));
            cudaMalloc(&x, len * WORDS * sizeof(u32));
            cudaMemcpy(x, data, len * WORDS * sizeof(u32), cudaMemcpyHostToDevice);

            mont256::Params* param_d;
            cudaMalloc(&param_d, sizeof(mont256::Params));
            cudaMemcpy(param_d, &param, sizeof(mont256::Params), cudaMemcpyHostToDevice);

            // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
            u32 log_p = 0u;
            
            if (timer) cudaEventRecord(start);

            // Each iteration performs a FFT round
            while (log_p < log_len) {

                // 1=>radix2, 2=>radix4, 3=>radix8, ...
                u32 deg = std::min(max_deg, log_len - log_p);

                assert(deg - 1 <= 10);

                dim3 block(1 << (deg - 1));
                dim3 grid(len >> deg);

                bellperson_kernel <WORDS> <<< grid, block, sizeof(u32) * WORDS * (1 << deg) >>> (x, y, pq_d, omegas_d, len, log_p, deg, max_deg, param_d);

                log_p += deg;

                u32 * tmp = x;
                x = y;
                y = tmp;
            }

            if (timer) {
                cudaEventRecord(end);
                cudaEventSynchronize(end);
                cudaEventElapsedTime(&milliseconds, start, end);
            }

            cudaMemcpy(data, x, len * WORDS * sizeof(u32), cudaMemcpyDeviceToHost);

            cudaFree(pq_d);
            cudaFree(omegas_d);
            cudaFree(y);
            cudaFree(x);

        }
    };
}