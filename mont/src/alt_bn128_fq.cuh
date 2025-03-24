#pragma once
#include "field.cuh"
#include "field2.cuh"

namespace alt_bn128_fq
{
    using Number = mont::Number<8>;
    using mont::u32;

    // 21888242871839275222246405745257275088548364400416034343698204186575808495617
    namespace device_constants {
    const __device__ Number m = BIG_INTEGER_CHUNKS8(0x30644e72, 0xe131a029, 0xb85045b6, 0x8181585d, 0x2833e848, 0x79b97091, 0x43e1f593, 0xf0000001);
    const __device__ Number mm2 = BIG_INTEGER_CHUNKS8(0x60c89ce5, 0xc2634053, 0x70a08b6d, 0x0302b0ba, 0x5067d090, 0xf372e122, 0x87c3eb27, 0xe0000002);
    const __device__ Number m_sub2 = BIG_INTEGER_CHUNKS8(0x30644e72, 0xe131a029, 0xb85045b6, 0x8181585d, 0x2833e848, 0x79b97091, 0x43e1f593, 0xefffffff);
    const __device__ Number r_mod = BIG_INTEGER_CHUNKS8(0xe0a77c1, 0x9a07df2f, 0x666ea36f, 0x7879462e, 0x36fc7695, 0x9f60cd29, 0xac96341c, 0x4ffffffb);
    const __device__ Number r2_mod = BIG_INTEGER_CHUNKS8(0x216d0b1, 0x7f4e44a5, 0x8c49833d, 0x53bb8085, 0x53fe3ab1, 0xe35c59e3, 0x1bb8e645, 0xae216da7);
    };
    namespace host_constants {
    const Number m = BIG_INTEGER_CHUNKS8(0x30644e72, 0xe131a029, 0xb85045b6, 0x8181585d, 0x2833e848, 0x79b97091, 0x43e1f593, 0xf0000001);
    const Number mm2 = BIG_INTEGER_CHUNKS8(0x60c89ce5, 0xc2634053, 0x70a08b6d, 0x0302b0ba, 0x5067d090, 0xf372e122, 0x87c3eb27, 0xe0000002);
    const Number m_sub2 = BIG_INTEGER_CHUNKS8(0x30644e72, 0xe131a029, 0xb85045b6, 0x8181585d, 0x2833e848, 0x79b97091, 0x43e1f593, 0xefffffff);
    const Number r_mod = BIG_INTEGER_CHUNKS8(0xe0a77c1, 0x9a07df2f, 0x666ea36f, 0x7879462e, 0x36fc7695, 0x9f60cd29, 0xac96341c, 0x4ffffffb);
    const Number r2_mod = BIG_INTEGER_CHUNKS8(0x216d0b1, 0x7f4e44a5, 0x8c49833d, 0x53bb8085, 0x53fe3ab1, 0xe35c59e3, 0x1bb8e645, 0xae216da7);
    };
    const u32 m_prime = 4026531839;

    struct Params
    {
        static const mont::usize LIMBS = 8;
        static const __host__ __device__ __forceinline__ Number m()
        {
    #ifdef __CUDA_ARCH__
        return device_constants::m;
    #else
        return host_constants::m;
    #endif
        }
        // m * 2
        static const __host__ __device__ __forceinline__ Number mm2()
        {
    #ifdef __CUDA_ARCH__
        return device_constants::mm2;
    #else
        return host_constants::mm2;
    #endif
        }
        // m - 2
        static const __host__ __device__ __forceinline__ Number m_sub2()
        {
    #ifdef __CUDA_ARCH__
        return device_constants::m_sub2;
    #else
        return host_constants::m_sub2;
    #endif
        }
        // m' = -m^(-1) mod b where b = 2^32
        static const u32 m_prime = m_prime;
        // r_mod = R mod m,
        static const __host__ __device__ __forceinline__ Number r_mod()
        {
    #ifdef __CUDA_ARCH__
        return device_constants::r_mod;
    #else
        return host_constants::r_mod;
    #endif
        }
        // r2_mod = R^2 mod m
        static const __host__ __device__ __forceinline__ Number r2_mod()
        {

    #ifdef __CUDA_ARCH__
        return device_constants::r2_mod;
    #else
        return host_constants::r2_mod;
    #endif
        }
    };

    using Element = mont::Element<Params>;
}

namespace alt_bn128_fq2
{
    using Number = mont::Number<8>;
    using mont::u32;

    // 21888242871839275222246405745257275088696311157297823662689037894645226208582
    const __device__ Number non_residue_device = BIG_INTEGER_CHUNKS8(0xa6d7254, 0x15043de8, 0x2a00b873, 0x3e2f4c49, 0x789db727, 0xbd66169f, 0x4b194dfd, 0xeec2024a);
    const Number non_residue = BIG_INTEGER_CHUNKS8(0xa6d7254, 0x15043de8, 0x2a00b873, 0x3e2f4c49, 0x789db727, 0xbd66169f, 0x4b194dfd, 0xeec2024a);

    struct Params
    {
        static const __host__ __device__ __forceinline__ alt_bn128_fq::Element non_residue()
        {
    #ifdef __CUDA_ARCH__
        return alt_bn128_fq::Element(alt_bn128_fq2::non_residue_device);
    #else
        return alt_bn128_fq::Element(alt_bn128_fq2::non_residue);
    #endif
        }
    };

    using Element = mont::Element2<alt_bn128_fq::Element, Params>;
}
