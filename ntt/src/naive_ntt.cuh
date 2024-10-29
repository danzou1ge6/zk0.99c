#pragma once

#include "ntt.cuh"

namespace ntt {
    template<typename Field>
    __global__ void naive(u32_E* data, u32 len, u32_E* roots, u32 stride) {
        static const mont::usize WORDS = Field::LIMBS;
        
        u32 id = (blockDim.x * blockIdx.x + threadIdx.x);
        if (id << 1 >= len) return;

        u64 offset = id % stride;
        u64 pos = ((id - offset) << 1) + offset;

        auto a = Field::load(data + (pos * WORDS));
        auto b = Field::load(data + ((pos + stride) * WORDS));
        auto w = Field::load(roots + (offset * len / (stride << 1)) * WORDS);

        b = b * w;
        
        auto tmp = a;
        a = a + b;
        b = tmp - b;
        
        a.store(data + (pos * WORDS));
        b.store(data + ((pos + stride) * WORDS));
    }


    template<typename Field>
    class naive_ntt : public best_ntt {
        const static usize WORDS = Field::LIMBS;
        using Number = mont::Number<WORDS>;

        Field unit;
        bool debug;
        u32 *roots, *roots_d;
        const u32 log_len;
        const u64 len;

        public:
        float milliseconds = 0;

        naive_ntt(const u32* omega, u32 log_len, bool debug) : log_len(log_len), len(1 << log_len), debug(debug) {
            cudaError_t first_err = cudaSuccess;
            
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

            CUDA_CHECK(cudaHostAlloc(&roots, ((u64)len / 2) * WORDS * sizeof(u32_E), cudaHostAllocDefault));

            gen_roots_cub<Field> gen;
            CUDA_CHECK(gen(roots, len / 2, unit));

            if (first_err != cudaSuccess) {
                std::cerr << "error occurred during gen_roots" << std::endl;
                throw cudaGetErrorString(first_err);
            }
        }

        ~naive_ntt() override {
            cudaFreeHost(roots);
            if (this->on_gpu) clean_gpu();
        }

        cudaError_t to_gpu(cudaStream_t stream = 0) override {
            std::unique_lock<std::shared_mutex> wlock(this->mtx);
            if (this->on_gpu) return cudaSuccess;

            cudaError_t first_err = cudaSuccess;

            CUDA_CHECK(cudaMalloc(&roots_d, len / 2 * WORDS * sizeof(u32)));
            CUDA_CHECK(cudaMemcpy(roots_d, roots, ((u64)len / 2) * WORDS * sizeof(u32_E), cudaMemcpyHostToDevice));

            if (first_err != cudaSuccess) {
                CUDA_CHECK(cudaFree(roots_d));
            } else {
                this->on_gpu = true;
            }
            return first_err;
        }

        cudaError_t clean_gpu(cudaStream_t stream = 0) override {
            std::unique_lock<std::shared_mutex> wlock(this->mtx);
            if (!this->on_gpu) return cudaSuccess;
            cudaError_t first_err = cudaSuccess;
            CUDA_CHECK(cudaFree(roots_d));
            this->on_gpu = false;
            return first_err;
        }

        cudaError_t ntt(u32 * data, cudaStream_t stream = 0, u32 start_n = 0, u32 **dev_ptr = nullptr) override {
            cudaError_t first_err = cudaSuccess;

            cudaEvent_t start, end;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&end));

            std::shared_lock<std::shared_mutex> rlock(this->mtx);
            {
                while(!this->on_gpu) {
                    rlock.unlock();
                    CUDA_CHECK(to_gpu());
                    rlock.lock();
                }
            }

            u32_E *data_d;
            CUDA_CHECK(cudaMalloc(&data_d, len * WORDS * sizeof(u32)));
            CUDA_CHECK(cudaMemcpy(data_d, data, len * WORDS * sizeof(u32), cudaMemcpyHostToDevice));

            if (debug) {
                dim3 block(768);
                dim3 grid((len - 1) / block.x + 1);
                number_to_element <Field> <<< grid, block >>> (data_d, len);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventRecord(start));
            }

            dim3 block(1024);
            dim3 grid((((u64) len) * WORDS - 1) / block.x + 1);
            rearrange<WORDS> <<<grid, block >>> (data_d, log_len);
            CUDA_CHECK(cudaGetLastError());

            dim3 ntt_block(768);
            dim3 ntt_grid(((len >> 1) - 1) / ntt_block.x + 1);

            for (u32 stride = 1; stride < len; stride <<= 1) {
                naive <Field> <<<ntt_grid, ntt_block>>> (data_d, len, roots_d, stride);
                CUDA_CHECK(cudaGetLastError());
            }

            if (debug) {
                CUDA_CHECK(cudaEventRecord(end));
                CUDA_CHECK(cudaEventSynchronize(end));
                CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
                dim3 block(768);
                dim3 grid((len - 1) / block.x + 1);
                element_to_number <Field> <<< grid, block >>> (data_d, len);
                CUDA_CHECK(cudaGetLastError());
            }

            rlock.unlock();

            CUDA_CHECK(cudaMemcpy(data, data_d, ((u64)len) * WORDS * sizeof(u32), cudaMemcpyDeviceToHost));

            CUDA_CHECK(cudaFree(data_d));

            if (debug) {
                CUDA_CHECK(clean_gpu());
            }
            
            return first_err;
        }
    };
}