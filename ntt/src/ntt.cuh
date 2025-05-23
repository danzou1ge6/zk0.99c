#pragma once

// #include "../../mont/src/mont.cuh"
#include "../../mont/src/field.cuh"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <cuda/barrier>
#include <iostream>
#include <shared_mutex>
#include <semaphore>
#include <bit>

#define CUDA_CHECK(call)                                                                                             \
{                                                                                                                    \
    cudaError_t err = call;                                                                                          \
    if (err != cudaSuccess) {                                                                                        \
        if (first_err == cudaSuccess) first_err = err;                                                                                \
        std::cerr << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: " << cudaGetErrorString(err) << std::endl; \
    }                                                                                                                \
}

namespace ntt {
    using mont::u32;
    using mont::u64;
    using mont::usize;
    typedef u32 u32_E;
    typedef u32 u32_N;
    const u32 MAX_NTT_INSTANCES = 1024; // 1024 should be big enough for typically there will be only 64 threads

    class best_ntt {
        protected:
        std::counting_semaphore<MAX_NTT_INSTANCES> sem;
        std::shared_mutex mtx; // lock for on_gpu data
        public:
        bool on_gpu = false;
        best_ntt(u32 max_instance = 1) : sem(std::min(max_instance, MAX_NTT_INSTANCES)) {}
        virtual cudaError_t ntt(u32 * data, cudaStream_t stream = 0, u32 start_n = 0, bool data_on_gpu = false) = 0;
        virtual ~best_ntt() = default;
        virtual cudaError_t to_gpu(cudaStream_t stream = 0) = 0;
        virtual cudaError_t clean_gpu(cudaStream_t stream = 0) = 0;
    };

    template <typename Field>
    class gen_roots_cub {
        public:
        struct get_iterator_to_range {
            __host__ __device__ __forceinline__ auto operator()(u32 index) {
                return thrust::make_constant_iterator(input_d[index]);
            }
            Field *input_d;
        };

        struct get_ptr_to_range {
            __host__ __device__ __forceinline__ auto operator()(u32 index) {
                return output_d + offsets_d[index];
            }
            Field *output_d;
            u32 *offsets_d;
        };

        struct get_run_length {
            __host__ __device__ __forceinline__ auto operator()(u32 index) {
                return offsets_d[index + 1] - offsets_d[index];
            }
            uint32_t *offsets_d;
        };

        struct mont_mul {
            __device__ __forceinline__ Field operator()(const Field &a, const Field &b) {
                return a * b;
            }
        };


        __host__ __forceinline__ cudaError_t operator() (u32_E * roots, u32 len, Field &unit) {
            cudaError_t first_err = cudaSuccess;

            if (len == 0) return first_err;
            if (len == 1) {
                Field::one().store(roots);
                return first_err;
            }
            
            const u32 num_ranges = 2;

            Field input[num_ranges] = {Field::one(), unit}; // {one, unit}

            Field * input_d;
            CUDA_CHECK(cudaMalloc(&input_d, num_ranges * sizeof(Field)));
            CUDA_CHECK(cudaMemcpy(input_d, input, num_ranges * sizeof(Field), cudaMemcpyHostToDevice));

            u32 offset[] = {0, 1, len};
            u32 * offset_d;
            CUDA_CHECK(cudaMalloc(&offset_d, (num_ranges + 1) * sizeof(u32)));
            CUDA_CHECK(cudaMemcpy(offset_d, offset, (num_ranges + 1) * sizeof(u32), cudaMemcpyHostToDevice));

            Field * output_d;
            CUDA_CHECK(cudaMalloc(&output_d, len * sizeof(Field)));

            // Returns a constant iterator to the element of the i-th run
            thrust::counting_iterator<u32> iota(0);
            auto iterators_in = thrust::make_transform_iterator(iota, get_iterator_to_range{input_d});

            // Returns the run length of the i-th run
            auto sizes = thrust::make_transform_iterator(iota, get_run_length{offset_d});

            // Returns pointers to the output range for each run
            auto ptrs_out = thrust::make_transform_iterator(iota, get_ptr_to_range{output_d, offset_d});

            // Determine temporary device storage requirements
            void *tmp_storage_d = nullptr;
            size_t temp_storage_bytes = 0;

            CUDA_CHECK(cub::DeviceCopy::Batched(tmp_storage_d, temp_storage_bytes, iterators_in, ptrs_out, sizes, num_ranges));

            // Allocate temporary storage
            CUDA_CHECK(cudaMalloc(&tmp_storage_d, temp_storage_bytes));

            // Run batched copy algorithm (used to perform runlength decoding)
            // output_d       <-- [one, unit, unit, ... , unit]
            CUDA_CHECK(cub::DeviceCopy::Batched(tmp_storage_d, temp_storage_bytes, iterators_in, ptrs_out, sizes, num_ranges));

            CUDA_CHECK(cudaFree(tmp_storage_d));
            CUDA_CHECK(cudaFree(input_d));
            CUDA_CHECK(cudaFree(offset_d));
            
            tmp_storage_d = nullptr;
            temp_storage_bytes = 0;

            mont_mul op;

            CUDA_CHECK(cub::DeviceScan::InclusiveScan(tmp_storage_d, temp_storage_bytes, output_d, op, len));
            CUDA_CHECK(cudaMalloc(&tmp_storage_d, temp_storage_bytes));
            CUDA_CHECK(cub::DeviceScan::InclusiveScan(tmp_storage_d, temp_storage_bytes, output_d, op, len));

            CUDA_CHECK(cudaMemcpy(roots, output_d, len * sizeof(Field), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(output_d));
            CUDA_CHECK(cudaFree(tmp_storage_d));
            
            return first_err;
        }
    };

    template<typename Field>
    __global__ void element_to_number(u32* data, u32 len) {
        u64 index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= len) return;
        Field::load(data + index * Field::LIMBS).to_number().store(data + index * Field::LIMBS);
    }

    template<typename Field>
    __global__ void number_to_element(u32* data, u32 len) {
        using Number = mont::Number<Field::LIMBS>;
        u64 index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= len) return;
        Field::from_number(Number::load(data + index * Field::LIMBS)).store(data + index * Field::LIMBS);
    }
    
    template<usize WORDS>
    __global__ void rearrange(u32_E * data, u32 log_len) {
        auto max_grid = ((1ll << log_len) * WORDS - 1) / blockDim.x + 1;
        for (int bid = blockIdx.x; bid < max_grid; bid += gridDim.x) {
            u64 index = bid * (blockDim.x / WORDS) + threadIdx.x / WORDS;
            u32 word = threadIdx.x & (WORDS - 1);
            if (index >= 1 << log_len) continue;
            u64 rindex = (__brev(index) >> (32 - log_len));

            if (rindex >= index) continue;

            uint4 tmp = reinterpret_cast<uint4*>(data)[index * WORDS + word];
            reinterpret_cast<uint4*>(data)[index * WORDS + word] = reinterpret_cast<uint4*>(data)[rindex * WORDS + word];
            reinterpret_cast<uint4*>(data)[rindex * WORDS + word] = tmp;
        }
    }

    template <typename Field>
    __forceinline__ __device__ Field pow_lookup_constant(u32 exponent, const u32_E *omegas) {
        static const usize WORDS = Field::LIMBS;
        auto res = Field::one();
        u32 i = 0;
        while(exponent > 0) {
            if (exponent & 1) {
                res = res * Field::load(omegas + (i * WORDS));
            }
            exponent = exponent >> 1;
            i++;
        }
        return res;
    }

    static __host__ __device__ __forceinline__ constexpr u32 log2_int(u32 x) {
        u32 ans = 0;
        while (x >>= 1) ans++;
        return ans;
        // return 31 - std::countl_zero(x);
    }
}