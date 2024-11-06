#pragma once
#include "../../mont/src/field.cuh"
#include <cuda_runtime.h>
#include <cub/cub.cuh>

namespace poly {
    using mont::u32;
    using mont::u64;
    using mont::usize;
    using mont::load_exchange;
    using mont::store_exchange;
    
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

        a_val = load_exchange<Field, io_group>(a, [] (u32 lid) -> u64 {return lid;}, temp_storage);
        b_val = load_exchange<Field, io_group>(b, [] (u32 lid) -> u64 {return lid;}, temp_storage);

        auto res = a_val + b_val;

        store_exchange<Field, io_group>(res, dst, [] (u32 lid) -> u64 {return lid;}, temp_storage);
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

        a_val = load_exchange<Field, io_group>(a, [] (u32 lid) -> u64 {return lid;}, temp_storage);
        b_val = load_exchange<Field, io_group>(b, [] (u32 lid) -> u64 {return lid;}, temp_storage);

        auto res = a_val * b_val;

        store_exchange<Field, io_group>(res, dst, [] (u32 lid) -> u64 {return lid;}, temp_storage);
    }
};