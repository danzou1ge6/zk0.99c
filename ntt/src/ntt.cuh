#pragma once

#include "../../mont/src/mont.cuh"
#include <cuda_runtime.h>
#include <cassert>
#include <cub/cub.cuh>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <bit>
#include <cuda/barrier>
#include <iostream>

#define CUDA_CHECK(call)                                                                                             \
{                                                                                                                    \
    cudaError_t err = call;                                                                                          \
    if (err != cudaSuccess) {                                                                                        \
        if (success) first_err = err;                                                                                \
        std::cerr << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: " << cudaGetErrorString(err) << std::endl; \
        success = false;                                                                                             \
    }                                                                                                                \
}

namespace ntt {
    typedef uint u32;
    typedef unsigned long long u64;
    typedef uint u32_E;
    typedef uint u32_N;

    class best_ntt {
        public:
        bool on_gpu = false;
        virtual cudaError_t ntt(u32 * data) = 0;
        virtual ~best_ntt() {
            if (on_gpu) clean_gpu();
        }
        virtual cudaError_t to_gpu() {return cudaSuccess;}
        virtual cudaError_t clean_gpu() {return cudaSuccess;}
    };

    template <u32 WORDS>
    class gen_roots_cub {
        public:
        struct element_pack {
            u32 data[WORDS];
        };
        struct get_iterator_to_range {
            __host__ __device__ __forceinline__ auto operator()(u32 index) {
                return thrust::make_constant_iterator(input_d[index]);
            }
            element_pack *input_d;
        };

        struct get_ptr_to_range {
            __host__ __device__ __forceinline__ auto operator()(u32 index) {
                return output_d + offsets_d[index];
            }
            element_pack *output_d;
            u32 *offsets_d;
        };

        struct get_run_length {
            __host__ __device__ __forceinline__ auto operator()(u32 index) {
                return offsets_d[index + 1] - offsets_d[index];
            }
            uint32_t *offsets_d;
        };
        struct mont_mul {
            __device__ __forceinline__ element_pack operator()(const element_pack &a, const element_pack &b) {
                auto env = mont256::Env(*param_d);
                element_pack res;
                auto a_ = mont256::Element::load(a.data);
                auto b_ = mont256::Element::load(b.data);
                auto c_ = env.mul(a_, b_);
                c_.store(res.data);
                return res;
            }
            mont256::Params* param_d;
        };


        __host__ __forceinline__ cudaError_t operator() (u32_E * roots, u32 len, mont256::Element &unit, const mont256::Params &param) {
            bool success = true;
            cudaError_t first_err = cudaSuccess;
            
            auto env_host = mont256::Env::host_new(param);

            const u32 num_ranges = 2;

            element_pack input[num_ranges]; // {one, unit}
            env_host.one().store(input[0].data);
            unit.store(input[1].data);
            
            element_pack * input_d;
            if (success) CUDA_CHECK(cudaMalloc(&input_d, num_ranges * sizeof(element_pack)));
            if (success) CUDA_CHECK(cudaMemcpy(input_d, input, num_ranges * sizeof(element_pack), cudaMemcpyHostToDevice));

            u32 offset[] = {0, 1, len};
            u32 * offset_d;
            if (success) CUDA_CHECK(cudaMalloc(&offset_d, (num_ranges + 1) * sizeof(u32)));
            if (success) CUDA_CHECK(cudaMemcpy(offset_d, offset, (num_ranges + 1) * sizeof(u32), cudaMemcpyHostToDevice));

            element_pack * output_d;
            if (success) CUDA_CHECK(cudaMalloc(&output_d, len * sizeof(element_pack)));

            // Returns a constant iterator to the element of the i-th run
            thrust::counting_iterator<uint32_t> iota(0);
            auto iterators_in = thrust::make_transform_iterator(iota, get_iterator_to_range{input_d});

            // Returns the run length of the i-th run
            auto sizes = thrust::make_transform_iterator(iota, get_run_length{offset_d});

            // Returns pointers to the output range for each run
            auto ptrs_out = thrust::make_transform_iterator(iota, get_ptr_to_range{output_d, offset_d});

            // Determine temporary device storage requirements
            void *tmp_storage_d = nullptr;
            size_t temp_storage_bytes = 0;

            if (success) CUDA_CHECK(cub::DeviceCopy::Batched(tmp_storage_d, temp_storage_bytes, iterators_in, ptrs_out, sizes, num_ranges));

            // Allocate temporary storage
            if (success) CUDA_CHECK(cudaMalloc(&tmp_storage_d, temp_storage_bytes));

            // Run batched copy algorithm (used to perform runlength decoding)
            // output_d       <-- [one, unit, unit, ... , unit]
            if (success) CUDA_CHECK(cub::DeviceCopy::Batched(tmp_storage_d, temp_storage_bytes, iterators_in, ptrs_out, sizes, num_ranges));

            CUDA_CHECK(cudaFree(tmp_storage_d));
            CUDA_CHECK(cudaFree(input_d));
            CUDA_CHECK(cudaFree(offset_d));
            
            tmp_storage_d = nullptr;
            temp_storage_bytes = 0;

            mont256::Params *param_d;
            if (success) CUDA_CHECK(cudaMalloc(&param_d, sizeof(mont256::Params)));
            if (success) CUDA_CHECK(cudaMemcpy(param_d, &param, sizeof(mont256::Params), cudaMemcpyHostToDevice));

            auto op = mont_mul{param_d};

            if (success) CUDA_CHECK(cub::DeviceScan::InclusiveScan(tmp_storage_d, temp_storage_bytes, output_d, op, len));
            if (success) CUDA_CHECK(cudaMalloc(&tmp_storage_d, temp_storage_bytes));
            if (success) CUDA_CHECK(cub::DeviceScan::InclusiveScan(tmp_storage_d, temp_storage_bytes, output_d, op, len));

            if (success) CUDA_CHECK(cudaMemcpy(roots, output_d, len * sizeof(element_pack), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(output_d));
            CUDA_CHECK(cudaFree(tmp_storage_d));
            CUDA_CHECK(cudaFree(param_d));
            
            return first_err;
        }
    };

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
    
    template<u64 WORDS>
    __global__ void rearrange(u32_E * data, u32 log_len) {
        u32 index = blockIdx.x * (blockDim.x / WORDS) + threadIdx.x / WORDS;
        u32 word = threadIdx.x & (WORDS - 1);
        if (index >= 1 << log_len) return;
        u32 rindex = (__brev(index) >> (32 - log_len));
        
        if (rindex >= index) return;

        u32 tmp = data[index * WORDS + word];
        data[index * WORDS + word] = data[rindex * WORDS + word];
        data[rindex * WORDS + word] = tmp;
    }

        
    template <u32 WORDS>
    __forceinline__ __device__ mont256::Element pow_lookup_constant(u32 exponent, mont256::Env &env, const u32_E *omegas) {
        auto res = env.one();
        u32 i = 0;
        while(exponent > 0) {
            if (exponent & 1) {
                res = env.mul(res, mont256::Element::load(omegas + (i * WORDS)));
            }
            exponent = exponent >> 1;
            i++;
        }
        return res;
    }
}