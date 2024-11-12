#pragma once
#include "self_sort_in_place_ntt.cuh"
#include <cassert>

namespace ntt {

    template <typename Field>
    class cooley_turkey_ntt : public best_ntt {
        const static usize WORDS = Field::LIMBS;
        using Number = mont::Number<WORDS>;
        
        u32 max_deg;
        const int log_len;
        const u64 len;
        Field unit;
        bool debug;
        u32_E *roots, *roots_d;
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
        u32 max_threads_stage_log = 8;
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

            
            CUDA_CHECK(cudaHostAlloc(&roots, (u64)len / 2 * WORDS * sizeof(u32_E), cudaHostAllocDefault));
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

                u32 group_num = std::min((int)(len / (1 << deg)), 1 << (max_threads_stage_log - (deg - 1)));

                u32 block_sz = (1 << (deg - 1)) * group_num;
                assert(block_sz <= (1 << max_threads_stage_log));
                u32 block_num = len / 2 / block_sz;
                assert(block_num * 2 * block_sz == len);

                dim3 block(block_sz);
                dim3 grid(block_num);

                auto kernel = (log_stride == log_len - 1 && (process && (!inverse))) ? 
                SSIP_NTT_stage1_warp_no_twiddle <Field, true> : SSIP_NTT_stage1_warp_no_twiddle <Field, false>;

                u32 shared_size = (sizeof(u32) * ((1 << deg) + 1) * WORDS) * group_num;
                        
                kernel <<< grid, block, shared_size, stream >>>(x, log_len, log_stride, deg, 1 << (deg - 1), roots_d, zeta_d, start_n);
                        
                CUDA_CHECK(cudaGetLastError());

                log_stride -= deg;
            }

            dim3 block(1024);
            dim3 grid(std::min(1024ul, (((u64) len) * WORDS / 4 - 1) / block.x + 1));
            rearrange<WORDS / 4> <<<grid, block >>> (x, log_len);

            // // manually tackle log_len == 1 case because the stage2 kernel won't run
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