#pragma once

#include "ntt.cuh"

namespace ntt {

    /*
    * FFT algorithm is inspired from: http://www.bealto.com/gpu-fft_group-1.html
    */
    template <u32 WORDS>
    __global__ void bellperson_kernel(u32_E * x, // Source buffer
                        u32_E * y, // Destination buffer
                        const u32_E * omegas,
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

        x += ((u64) index) * WORDS;
        y += ((u64) (((index - k) << deg) + k)) * WORDS;

        u32 count = 1 << deg; // 2^deg
        u32 counth = count >> 1; // Half of count

        u32 counts = count / lsize * lid;
        u32 counte = counts + count / lsize;

        // Compute powers of twiddle
        auto twiddle = pow_lookup_constant<WORDS>((n >> lgp >> deg) * k, env, omegas);

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
    class bellperson_ntt : public best_ntt {
        const u32 max_deg;
        const u32 log_len;
        const u64 len;
        mont256::Params param;
        mont256::Element unit;
        bool debug;
        u32_E *pq; // Precalculated values for radix degrees up to `max_deg`
        u32_E *omegas, *omegas_d; // Precalculated values for [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]
        u32_E* pq_d;
        mont256::Params* param_d;

        public:
        float milliseconds = 0;
        bellperson_ntt(const mont256::Params param, const u32* omega, u32 log_len, bool debug) : param(param), log_len(log_len), max_deg(std::min(8u, log_len)), len(1 << log_len), debug(debug) {
            // Precalculate:
            auto env = mont256::Env::host_new(param);

            if (debug) {
                // unit = qpow(omega, (P - 1ll) / len)
                unit = env.host_from_number(mont256::Number::load(omega));
                auto exponent = mont256::Number::load(param.m);
                auto one = mont256::Number::zero();
                one.c0 = 1;
                exponent = exponent.host_sub(one);
                exponent = exponent.slr(log_len);
                unit = env.host_pow(unit, exponent);
            } else {
                unit = mont256::Element::load(omega);
            }

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

        ~bellperson_ntt() override {
            free(pq);
            free(omegas);
        }

        cudaError_t to_gpu() override {
            if (this->on_gpu) return cudaSuccess;

            bool success = true;
            cudaError_t first_err = cudaSuccess;

            CUDA_CHECK(cudaMalloc(&pq_d, (1 << max_deg >> 1) * sizeof(u32_E) * WORDS))
            CUDA_CHECK(cudaMalloc(&omegas_d, 32 * sizeof(u32_E) * WORDS))
            CUDA_CHECK(cudaMalloc(&param_d, sizeof(mont256::Params)));

            CUDA_CHECK(cudaMemcpy(pq_d, pq, (1 << max_deg >> 1) * sizeof(u32_E) * WORDS, cudaMemcpyHostToDevice))
            CUDA_CHECK(cudaMemcpy(omegas_d, omegas, 32 * sizeof(u32_E) * WORDS, cudaMemcpyHostToDevice))
            CUDA_CHECK(cudaMemcpy(param_d, &param, sizeof(mont256::Params), cudaMemcpyHostToDevice));

            if (!success) {
                CUDA_CHECK(cudaFree(pq_d));
                CUDA_CHECK(cudaFree(omegas_d));
                CUDA_CHECK(cudaFree(param_d));
            } else {
                this->on_gpu = true;
            }
            return first_err;
        }

        cudaError_t clean_gpu() override {
            if (!this->on_gpu) return cudaSuccess;
            bool success = true;
            cudaError_t first_err = cudaSuccess;

            CUDA_CHECK(cudaFree(pq_d));
            CUDA_CHECK(cudaFree(omegas_d));
            CUDA_CHECK(cudaFree(param_d));

            this->on_gpu = false;
            return first_err;
        }

        cudaError_t ntt(u32 * data) override {
            bool success = true;
            cudaError_t first_err = cudaSuccess;

            cudaEvent_t start, end;
            if (success) CUDA_CHECK(cudaEventCreate(&start));
            if (success) CUDA_CHECK(cudaEventCreate(&end));
         
            if (!this->on_gpu) if (success) CUDA_CHECK(to_gpu());

            u32 * x, * y;
            if (success) CUDA_CHECK(cudaMalloc(&y, len * WORDS * sizeof(u32)));
            if (success) CUDA_CHECK(cudaMalloc(&x, len * WORDS * sizeof(u32)));
            if (success) CUDA_CHECK(cudaMemcpy(x, data, len * WORDS * sizeof(u32), cudaMemcpyHostToDevice));

            // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
            u32 log_p = 0u;
            
            if (debug) {
                dim3 block(768);
                dim3 grid((len - 1) / block.x + 1);
                if (success) number_to_element <WORDS> <<< grid, block >>> (x, len, param_d);
                if (success) CUDA_CHECK(cudaGetLastError());
                if (success) CUDA_CHECK(cudaEventRecord(start));
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
            
                if (success) CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size));

                if (success) kernel <<< grid, block, shared_size >>> (x, y, omegas_d, pq_d, len, log_p, deg, max_deg, param_d);
                
                if (success) CUDA_CHECK(cudaGetLastError());

                log_p += deg;

                u32 * tmp = x;
                x = y;
                y = tmp;
            }

            if (debug) {
                if (success) CUDA_CHECK(cudaEventRecord(end));
                if (success) CUDA_CHECK(cudaEventSynchronize(end));
                if (success) CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
                dim3 block(768);
                dim3 grid((len - 1) / block.x + 1);
                if (success) element_to_number <WORDS> <<< grid, block >>> (x, len, param_d);
                if (success) CUDA_CHECK(cudaGetLastError());
            }

            if (success) CUDA_CHECK(cudaMemcpy(data, x, len * WORDS * sizeof(u32), cudaMemcpyDeviceToHost));

            CUDA_CHECK(cudaFree(y));
            CUDA_CHECK(cudaFree(x));

            if (debug) {
                CUDA_CHECK(clean_gpu());
            }

            return first_err;
        }
    };
}