#pragma once

#include "../../mont/src/mont.cuh"
#include <cuda_runtime.h>

namespace NTT {
    typedef u_int32_t u32;
    
    template<u32 WORDS>
    class naive_ntt {
        mont256::Params param;
        mont256::Element unit;
        bool timer;

        __global__ void rearrange(u32 * data, uint2 * reverse, u32 len) {
            u32 index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= len) return;
            uint2 r = reverse[index];
            
            #pragma unroll
            for (u32 i = 0; i < WORDS; i++) {
                u32 tmp = data[r.x * WORDS + i];
                data[r.x * WORDS + i] = data[r.y * WORDS + i];
                data[r.y * WORDS + i] = tmp;
            }
        }

        __global__ void naive(u32* data[], u32 len, u32* roots[], u32 stride, mont256::Params &param) {
            auto env = mont256::Env(param);

            u32 id = (blockDim.x * blockIdx.x + threadIdx.x);
            if (id << 1 >= len) return;

            u32 offset = id & (stride - 1);
            u32 pos = (id << 1) - offset;

            auto a = mont256::Element::load(data + (pos * WORDS));
            auto b = mont256::Element::load(data + ((pos + stride) * WORDS));
            auto w = mont256::Element::load(roots + (offset * len / (stride << 1)) * WORDS);

            b = env.mul(b, w);
            
            auto tmp = a;
            a = env.add_modulo(a, b);
            b = env.sub_modulo(tmp, b);
            
            a.store(data + (pos * WORDS));
            b.store(data + ((pos + stride) * WORDS));
        }
        
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

        void gen_roots(u32 * roots, u32 len, mont256::Element unit) {
            auto env = mont256::Env::host_new(param);
            roots[0] = 1;
            auto last_element = mont256::Element::load(roots);
            for (u32 i = WORDS; i < len * WORDS; i+= WORDS) {
                last_element = env.host_mul(last_element, unit);
                last_element.store(roots + i);
            }
        }

        public:
        float milliseconds = 0;

        naive_ntt() {
            // TODO : set param
        }

        void ntt(u32 * data, u32 log_len) {
            cudaEvent_t start, end;
            cudaEventCreate(&start);
            cudaEventCreate(&end);

            u32 len = 1 << log_len;
            uint2 * reverse, * reverse_d;
            reverse = (uint2 *) malloc(len * sizeof(uint2));
            u32 r_len = gen_reverse(log_len, reverse);
            cudaMalloc(&reverse_d, r_len * sizeof(uint2));
            cudaMemcpy(reverse_d, reverse, r_len * sizeof(uint2), cudaMemcpyHostToDevice);
            
            dim3 rearrange_block(768);
            dim3 rearrange_grid((r_len + rearrange_block.x - 1) / rearrange_block.x);

            u32 * data_d;
            cudaMalloc(&data_d, len * WORDS * sizeof(u32));
            cudaMemcpy(data_d, data, len * WORDS * sizeof(u32), cudaMemcpyHostToDevice);

            

            mont256::Element unit_cur; // TODO : set unit
            
            u32 * roots, * roots_d;
            roots = (u32 *) malloc(len * WORDS * sizeof(u32));
            cudaMalloc(&roots_d, len * WORDS * sizeof(u32));
            gen_roots(roots, len, unit_cur);
            cudaMemcpy(roots_d, roots, len * WORDS * sizeof(u32), cudaMemcpyHostToDevice);

            dim3 ntt_block(768);
            dim3 ntt_grid(((len >> 1) - 1) / ntt_block.x + 1);

            mont256::Params param_d;
            cudaMemcpy(&param_d, &param, sizeof(mont256::Params), cudaMemcpyHostToDevice);

            if (timer) cudaEventRecord(start);
            rearrange<<<rearrange_grid, rearrange_block>>>(data, reverse, r_len);

            for (u32 stride = 1; stride < len; stride <<= 1) {
                naive<<<ntt_grid, ntt_block>>>(data_d, len, roots_d, stride, param_d);
            }

            if (timer) {
                cudaEventRecord(end);
                cudaEventSynchronize(end);
                cudaEventElapsedTime(&milliseconds, start, end);
            }

            cudaMemcpy(data, data_d, len * WORDS * sizeof(u32), cudaMemcpyDeviceToHost);

            cudaFree(data_d);
            cudaFree(reverse_d);
            cudaFree(roots_d);
            free(roots);
            free(reverse);
        }
    };

    template<u32 WORDS>
    class bellperson_ntt {
        const u32 max_deg = 8u;
        const u32 log_len, len;
        mont256::Params param;
        mont256::Element unit;
        bool timer;
        u32 *pq; // Precalculated values for radix degrees up to `max_deg`
        u32 *omegas; // Precalculated values for [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]

        __forceinline__ __device__ mont256::Element pow_lookup(u32 *omegas, u32 exponent, mont256::Env &env) {
            auto res = mont256::Element::one();
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
        __global__ void bellperson_kernel(u32 * x, // Source buffer
                            u32 * y, // Destination buffer
                            u32 * pq, // Precalculated twiddle factors
                            u32 * omegas, // [omega, omega^2, omega^4, ...]
                            u32 n, // Number of elements
                            u32 lgp, // Log2 of `p` (Read more in the link above)
                            u32 deg, // 1=>radix2, 2=>radix4, 3=>radix8, ...
                            u32 max_deg, // Maximum degree supported, according to `pq` and `omegas`
                            mont256::Params &param)
        {

            // There can only be a single dynamic shared memory item, hence cast it to the type we need.
            extern __shared__ u32* u[];

            auto env = mont256::Env(param);

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
            auto twiddle = pow_lookup(omegas, (n >> lgp >> deg) * k, env);

            // TODO: change to u32 pow
            auto exponent = mont256::Number::zero();
            exponent.c0 = counts;
            auto tmp = env.pow(twiddle, exponent);

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
                    a.load(u + i0 * WORDS);
                    b.load(u + i1 * WORDS);
                    tmp = a;
                    a = env.add_modulo(a, b);
                    b = env.sub_modulo(tmp, b);
                    if (di != 0) {
                        w.load(pq + (di << rnd << pqshift) * WORDS);
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

        public:
        float milliseconds = 0;
        bellperson_ntt(u32 log_len, bool timer) : log_len(log_len), len(1 << log_len), timer(timer) {
            // TODO : set param and unit
            

            // Precalculate:
            auto env = mont256::Env(param);

            // TODO: unit = qpow(omega, (P - 1ll) / n);

            // pq: [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]

            pq = (u32 *) malloc((1 << max_deg >> 1) * sizeof(u32) * WORDS);
            memset(pq, 0, (1 << max_deg >> 1) * sizeof(u32) * WORDS);
            pq[0] = 1;
            auto exponent = mont256::Number::zero();
            exponent.c0 = len >> max_deg;
            auto twiddle = env.host_pow(unit, exponent);
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

            mont256::Params param_d;
            cudaMemcpy(&param_d, &param, sizeof(mont256::Params), cudaMemcpyHostToDevice);

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

                bellperson_kernel <<< grid, block, sizeof(u32) * WORDS * (1 << deg) >>>(x, y, pq_d, omegas_d, len, log_p, deg, max_deg, param_d);

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

            return x;
        }
    };
}