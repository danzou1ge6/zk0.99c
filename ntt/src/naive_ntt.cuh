#pragma once

#include "ntt.cuh"

namespace ntt {
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
    class naive_ntt : public best_ntt {
        mont256::Params *param_d, *param;
        mont256::Element unit;
        bool debug;
        u32_E *roots, *roots_d;
        const u32 log_len;
        const u64 len;

        void gen_roots(u32_E * roots, u32 len) {
            auto env = mont256::Env::host_new(*param);
            env.one().store(roots);
            auto last_element = mont256::Element::load(roots);
            for (u64 i = WORDS; i < ((u64)len) * WORDS; i+= WORDS) {
                last_element = env.host_mul(last_element, unit);
                last_element.store(roots + i);
            }
        }

        public:
        float milliseconds = 0;

        naive_ntt(const mont256::Params &param, const u32_N* omega, u32 log_len, bool debug) : log_len(log_len), len(1 << log_len), debug(debug) {
            bool success = true;
            cudaError_t first_err = cudaSuccess;

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

            CUDA_CHECK(cudaHostAlloc(&roots, ((u64)len / 2) * WORDS * sizeof(u32_E), cudaHostAllocDefault));
            CUDA_CHECK(cudaHostAlloc(&this->param, sizeof(mont256::Params), cudaHostAllocDefault));
            CUDA_CHECK(cudaMemcpy(this->param, &param, sizeof(mont256::Params), cudaMemcpyHostToHost));

            // gen_roots(roots, len);
            gen_roots_cub<WORDS> gen;
            CUDA_CHECK(gen(roots, len / 2, unit, *this->param));
            if (!success) {
                std::cerr << "error occurred during gen_roots" << std::endl;
                throw cudaGetErrorString(first_err);
            }
        }

        ~naive_ntt() override {
            cudaFreeHost(roots);
            cudaFreeHost(param);
            if (this->on_gpu) {
                cudaFree(roots_d);
                cudaFree(param_d);
            }
        }

        cudaError_t to_gpu() override {
            std::unique_lock<std::shared_mutex> wlock(this->mtx);
            if (this->on_gpu) return cudaSuccess;

            bool success = true;
            cudaError_t first_err = cudaSuccess;

            CUDA_CHECK(cudaMalloc(&roots_d, len / 2 * WORDS * sizeof(u32)));
            CUDA_CHECK(cudaMalloc(&param_d, sizeof(mont256::Params)));
            CUDA_CHECK(cudaMemcpy(roots_d, roots, ((u64)len / 2) * WORDS * sizeof(u32_E), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(param_d, param, sizeof(mont256::Params), cudaMemcpyHostToDevice));

            if (!success) {
                CUDA_CHECK(cudaFree(roots_d));
                CUDA_CHECK(cudaFree(param_d));
            } else {
                this->on_gpu = true;
            }
            return first_err;
        }

        cudaError_t clean_gpu() override {
            std::unique_lock<std::shared_mutex> wlock(this->mtx);
            if (!this->on_gpu) return cudaSuccess;
            bool success = true;
            cudaError_t first_err = cudaSuccess;
            CUDA_CHECK(cudaFree(roots_d));
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

            std::shared_lock<std::shared_mutex> rlock(this->mtx);
            if (success) {
                while(!this->on_gpu) {
                    rlock.unlock();
                    CUDA_CHECK(to_gpu());
                    rlock.lock();
                }
            }

            u32_E *data_d;
            if (success) CUDA_CHECK(cudaMalloc(&data_d, len * WORDS * sizeof(u32)));
            if (success) CUDA_CHECK(cudaMemcpy(data_d, data, len * WORDS * sizeof(u32), cudaMemcpyHostToDevice));

            if (debug) {
                dim3 block(768);
                dim3 grid((len - 1) / block.x + 1);
                if (success) number_to_element <WORDS> <<< grid, block >>> (data_d, len, param_d);
                if (success) CUDA_CHECK(cudaGetLastError());
                if (success) CUDA_CHECK(cudaEventRecord(start));
            }

            dim3 block(1024);
            dim3 grid((((u64) len) * WORDS - 1) / block.x + 1);
            if (success) rearrange<WORDS> <<<grid, block >>> (data_d, log_len);
            if (success) CUDA_CHECK(cudaGetLastError());

            dim3 ntt_block(768);
            dim3 ntt_grid(((len >> 1) - 1) / ntt_block.x + 1);

            for (u32 stride = 1; stride < len; stride <<= 1) {
                if (success) naive <WORDS> <<<ntt_grid, ntt_block>>> (data_d, len, roots_d, stride, param_d);
                if (success) CUDA_CHECK(cudaGetLastError());
            }

            if (debug) {
                if (success) CUDA_CHECK(cudaEventRecord(end));
                if (success) CUDA_CHECK(cudaEventSynchronize(end));
                if (success) CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
                dim3 block(768);
                dim3 grid((len - 1) / block.x + 1);
                if (success) element_to_number <WORDS> <<< grid, block >>> (data_d, len, param_d);
                if (success) CUDA_CHECK(cudaGetLastError());
            }

            rlock.unlock();

            if (success) CUDA_CHECK(cudaMemcpy(data, data_d, ((u64)len) * WORDS * sizeof(u32), cudaMemcpyDeviceToHost));

            CUDA_CHECK(cudaFree(data_d));

            if (debug) {
                CUDA_CHECK(clean_gpu());
            }
            
            return first_err;
        }
    };
}