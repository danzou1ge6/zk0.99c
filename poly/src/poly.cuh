#pragma once
#include "../../mont/src/field.cuh"
#include <cuda_runtime.h>
#include <cub/cub.cuh>

namespace poly {
    using mont::u32;
    using mont::u64;
    using mont::usize;
    static const u32 warp_size = 32;

    template <typename Field>
    __global__ void NaiveAdd(u32 * a, u32 * b, u32 * dst, u32 len) {
        u64 index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= len) return;
        auto a_val = Field::load(a + index * Field::LIMBS);
        auto b_val = Field::load(b + index * Field::LIMBS);
        (a_val + b_val).store(dst + index * Field::LIMBS);
    }

    template <typename Field>
    __global__ void NaiveMul(u32 * a, u32 * b, u32 * dst, u32 len) {
        u64 index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= len) return;
        auto a_val = Field::load(a + index * Field::LIMBS);
        auto b_val = Field::load(b + index * Field::LIMBS);
        (a_val * b_val).store(dst + index * Field::LIMBS);
    }

    template <typename Field, u32 io_group>
    __forceinline__ __device__ auto load_exchange(u32 * data, typename cub::WarpExchange<u32, io_group, io_group>::TempStorage temp_storage[]) -> Field {
        using WarpExchangeT = cub::WarpExchange<u32, io_group, io_group>;

        const static usize WORDS = Field::LIMBS;
        const u32 io_id = threadIdx.x & (io_group - 1);
        const u32 lid_start = threadIdx.x - io_id;
        const int warp_id = static_cast<int>(threadIdx.x) / io_group;

        u32 thread_data[io_group];
        #pragma unroll
        for (u64 i = lid_start; i != lid_start + io_group; i ++) {
            if (io_id < WORDS) {
                thread_data[i - lid_start] = data[i * WORDS + io_id];
            }
        }
        WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);
        __syncwarp();
        return Field::load(thread_data);
    }

    template <typename Field, u32 io_group>
    __forceinline__ __device__ void store_exchange(Field ans, u32 * dst, typename cub::WarpExchange<u32, io_group, io_group>::TempStorage temp_storage[]) {
        using WarpExchangeT = cub::WarpExchange<u32, io_group, io_group>;

        const static usize WORDS = Field::LIMBS;
        const u32 io_id = threadIdx.x & (io_group - 1);
        const u32 lid_start = threadIdx.x - io_id;
        const int warp_id = static_cast<int>(threadIdx.x) / io_group;

        u32 thread_data[io_group];
        ans.store(thread_data);
        WarpExchangeT(temp_storage[warp_id]).BlockedToStriped(thread_data, thread_data);
        __syncwarp();
        #pragma unroll
        for (u64 i = lid_start; i != lid_start + io_group; i ++) {
            if (io_id < WORDS) {
                dst[i * WORDS + io_id] = thread_data[i - lid_start];
            }
        }
    }

    template <typename Field, u32 io_group>
    __global__ void Add(u32 * a, u32 * b, u32 * dst, u32 len) {       
        using WarpExchangeT = cub::WarpExchange<u32, io_group, io_group>;
        extern __shared__ typename WarpExchangeT::TempStorage temp_storage[];

        const static usize WORDS = Field::LIMBS;

        u32 index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= len) return;

        a += ((u64)blockIdx.x * blockDim.x) * WORDS;
        b += ((u64)blockIdx.x * blockDim.x) * WORDS;
        dst += ((u64)blockIdx.x * blockDim.x) * WORDS;

        Field a_val, b_val;

        a_val = load_exchange<Field, io_group>(a, temp_storage);
        b_val = load_exchange<Field, io_group>(b, temp_storage);

        auto res = a_val + b_val;

        store_exchange<Field, io_group>(res, dst, temp_storage);
    }

    template <typename Field, u32 io_group>
    __global__ void Mul(u32 * a, u32 * b, u32 * dst, u32 len) {       
        using WarpExchangeT = cub::WarpExchange<u32, io_group, io_group>;
        extern __shared__ typename WarpExchangeT::TempStorage temp_storage[];

        const static usize WORDS = Field::LIMBS;

        u32 index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= len) return;

        a += ((u64)blockIdx.x * blockDim.x) * WORDS;
        b += ((u64)blockIdx.x * blockDim.x) * WORDS;
        dst += ((u64)blockIdx.x * blockDim.x) * WORDS;

        Field a_val, b_val;

        a_val = load_exchange<Field, io_group>(a, temp_storage);
        b_val = load_exchange<Field, io_group>(b, temp_storage);

        auto res = a_val * b_val;

        store_exchange<Field, io_group>(res, dst, temp_storage);
    }
};