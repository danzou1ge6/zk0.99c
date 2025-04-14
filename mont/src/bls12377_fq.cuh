#pragma once
#include "field.cuh"

// base field for BLS12-377
namespace bls12377_fq
{

    using Number = mont::Number<12>;
    using mont::u32;
    // 258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177
    // const auto params = Params {
    //     .m = BIG_INTEGER_CHUNKS12(0x1ae3a46, 0x17c510ea, 0xc63b05c0, 0x6ca1493b, 0x1a22d9f3, 0x00f5138f, 0x1ef3622f, 0xba094800, 0x170b5d44, 0x30000000, 0x8508c000, 0x00000001),
    //     .mm2 = BIG_INTEGER_CHUNKS12(0x35c748c, 0x2f8a21d5, 0x8c760b80, 0xd9429276, 0x3445b3e6, 0x01ea271e, 0x3de6c45f, 0x74129000, 0x2e16ba88, 0x60000001, 0x0a118000, 0x00000002),
    //     .r_mod = BIG_INTEGER_CHUNKS12(0x8d6661, 0xe2fdf49a, 0x4cf495bf, 0x803c84e8, 0x7b4e97b7, 0x6e7c6305, 0x9f7db3a9, 0x8a7d3ff2, 0x51409f83, 0x7fffffb1, 0x02cdffff, 0xffffff68),
    //     .r2_mod = BIG_INTEGER_CHUNKS12(0x6dfccb, 0x1e914b88, 0x837e92f0, 0x41790bf9, 0xbfdf7d03, 0x827dc3ac, 0x22a5f111, 0x62d6b46d, 0x0329fcaa, 0xb00431b1, 0xb786686c, 0x9400cd22),
    //     .m_prime = 4294967295
    // };
    namespace device_constants
    {
        // m = 258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177
        const __device__ Number m = BIG_INTEGER_CHUNKS12(0x1ae3a46, 0x17c510ea, 0xc63b05c0, 0x6ca1493b, 0x1a22d9f3, 0x00f5138f, 0x1ef3622f, 0xba094800, 0x170b5d44, 0x30000000, 0x8508c000, 0x00000001);
        const __device__ Number m_sub2 = BIG_INTEGER_CHUNKS12(0x1ae3a46, 0x17c510ea, 0xc63b05c0, 0x6ca1493b, 0x1a22d9f3, 0x00f5138f, 0x1ef3622f, 0xba094800, 0x170b5d44, 0x30000000, 0x8508bfff, 0xffffffff);
        const __device__ Number r_mod = BIG_INTEGER_CHUNKS12(0x8d6661, 0xe2fdf49a, 0x4cf495bf, 0x803c84e8, 0x7b4e97b7, 0x6e7c6305, 0x9f7db3a9, 0x8a7d3ff2, 0x51409f83, 0x7fffffb1, 0x02cdffff, 0xffffff68);
        const __device__ Number r2_mod = BIG_INTEGER_CHUNKS12(0x6dfccb, 0x1e914b88, 0x837e92f0, 0x41790bf9, 0xbfdf7d03, 0x827dc3ac, 0x22a5f111, 0x62d6b46d, 0x0329fcaa, 0xb00431b1, 0xb786686c, 0x9400cd22);
    }

    namespace host_constants
    {
        const Number m = BIG_INTEGER_CHUNKS12(0x1ae3a46, 0x17c510ea, 0xc63b05c0, 0x6ca1493b, 0x1a22d9f3, 0x00f5138f, 0x1ef3622f, 0xba094800, 0x170b5d44, 0x30000000, 0x8508c000, 0x00000001);
        const Number m_sub2 = BIG_INTEGER_CHUNKS12(0x1ae3a46, 0x17c510ea, 0xc63b05c0, 0x6ca1493b, 0x1a22d9f3, 0x00f5138f, 0x1ef3622f, 0xba094800, 0x170b5d44, 0x30000000, 0x8508bfff, 0xffffffff);
        const Number r_mod = BIG_INTEGER_CHUNKS12(0x8d6661, 0xe2fdf49a, 0x4cf495bf, 0x803c84e8, 0x7b4e97b7, 0x6e7c6305, 0x9f7db3a9, 0x8a7d3ff2, 0x51409f83, 0x7fffffb1, 0x02cdffff, 0xffffff68);
        const Number r2_mod = BIG_INTEGER_CHUNKS12(0x6dfccb, 0x1e914b88, 0x837e92f0, 0x41790bf9, 0xbfdf7d03, 0x827dc3ac, 0x22a5f111, 0x62d6b46d, 0x0329fcaa, 0xb00431b1, 0xb786686c, 0x9400cd22);
    }

    struct Params
    {
        static const mont::usize LIMBS = 12;
        static const __host__ __device__ __forceinline__ Number m()
        {
    #ifdef __CUDA_ARCH__
        return device_constants::m;
    #else
        return host_constants::m;
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
        static const u32 m_prime = 4294967295;
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