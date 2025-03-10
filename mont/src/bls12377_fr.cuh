#pragma once
#include "field.cuh"

// scalar field for BLS12-377
namespace bls12377_fr
{
    // 8444461749428370424248824938781546531375899335154063827935233455917409239041
    // const auto params = Params {
    //     .m = BIG_INTEGER_CHUNKS8(0x12ab655e, 0x9a2ca556, 0x60b44d1e, 0x5c37b001, 0x59aa76fe, 0xd0000001, 0x0a118000, 0x00000001),
    //     .mm2 = BIG_INTEGER_CHUNKS8(0x2556cabd, 0x34594aac, 0xc1689a3c, 0xb86f6002, 0xb354edfd, 0xa0000002, 0x14230000, 0x00000002),
    //     .r_mod = BIG_INTEGER_CHUNKS8(0xd4bda32, 0x2bbb9a9d, 0x16d81575, 0x512c0fee, 0x7257f50f, 0x6ffffff2, 0x7d1c7fff, 0xfffffff3),
    //     .r2_mod = BIG_INTEGER_CHUNKS8(0x11fdae7, 0xeff1c939, 0xa7cc008f, 0xe5dc8593, 0xcc2c27b5, 0x8860591f, 0x25d577ba, 0xb861857b),
    //     .m_prime = 4294967295
    // };

    using Number = mont::Number<8>;
    using mont::u32;

    namespace device_constants
    {
        // 8444461749428370424248824938781546531375899335154063827935233455917409239041
        const __device__ Number m = BIG_INTEGER_CHUNKS8(0x12ab655e, 0x9a2ca556, 0x60b44d1e, 0x5c37b001, 0x59aa76fe, 0xd0000001, 0x0a118000, 0x00000001);
        const __device__ Number m_sub2 = BIG_INTEGER_CHUNKS8(0x12ab655e, 0x9a2ca556, 0x60b44d1e, 0x5c37b001, 0x59aa76fe, 0xd0000001, 0x0a117fff, 0xffffffff);
        const __device__ Number r_mod = BIG_INTEGER_CHUNKS8(0xd4bda32, 0x2bbb9a9d, 0x16d81575, 0x512c0fee, 0x7257f50f, 0x6ffffff2, 0x7d1c7fff, 0xfffffff3);
        const __device__ Number r2_mod = BIG_INTEGER_CHUNKS8(0x11fdae7, 0xeff1c939, 0xa7cc008f, 0xe5dc8593, 0xcc2c27b5, 0x8860591f, 0x25d577ba, 0xb861857b);
    }

    namespace host_constants
    {
        const Number m = BIG_INTEGER_CHUNKS8(0x12ab655e, 0x9a2ca556, 0x60b44d1e, 0x5c37b001, 0x59aa76fe, 0xd0000001, 0x0a118000, 0x00000001);
        const Number m_sub2 = BIG_INTEGER_CHUNKS8(0x12ab655e, 0x9a2ca556, 0x60b44d1e, 0x5c37b001, 0x59aa76fe, 0xd0000001, 0x0a117fff, 0xffffffff);
        const Number r_mod = BIG_INTEGER_CHUNKS8(0xd4bda32, 0x2bbb9a9d, 0x16d81575, 0x512c0fee, 0x7257f50f, 0x6ffffff2, 0x7d1c7fff, 0xfffffff3);
        const Number r2_mod = BIG_INTEGER_CHUNKS8(0x11fdae7, 0xeff1c939, 0xa7cc008f, 0xe5dc8593, 0xcc2c27b5, 0x8860591f, 0x25d577ba, 0xb861857b);
    }

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