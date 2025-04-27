#pragma once
#include "field.cuh"

// scalar field for BLS12-381
namespace bls12381_fr
{
    // bls12381_fr
    // 52435875175126190479447740508185965837690552500527637822603658699938581184513
    // const auto params = mont256::Params {
    //     .m = BIG_INTEGER_CHUNKS8(0x73eda753, 0x299d7d48, 0x3339d808, 0x09a1d805, 0x53bda402, 0xfffe5bfe, 0xffffffff, 0x00000001),
    //     .r_mod = BIG_INTEGER_CHUNKS8(0x1824b159, 0xacc5056f, 0x998c4fef, 0xecbc4ff5, 0x5884b7fa, 0x00034802, 0x00000001, 0xfffffffe),
    //     .r2_mod = BIG_INTEGER_CHUNKS8(0x748d9d9, 0x9f59ff11, 0x05d31496, 0x7254398f, 0x2b6cedcb, 0x87925c23, 0xc999e990, 0xf3f29c6d),
    //     .m_prime = 4294967295
    // };

    using mont::u32;
    u32 m[8] = BIG_INTEGER_CHUNKS8(0x73eda753, 0x299d7d48, 0x3339d808, 0x09a1d805, 0x53bda402, 0xfffe5bfe, 0xffffffff, 0x00000001);
    u32 m_sub2[8] = BIG_INTEGER_CHUNKS8(0x73eda753, 0x299d7d48, 0x3339d808, 0x09a1d805, 0x53bda402, 0xfffe5bfe, 0xfffffffe, 0xffffffff);
    u32 r_mod[8] = BIG_INTEGER_CHUNKS8(0x1824b159, 0xacc5056f, 0x998c4fef, 0xecbc4ff5, 0x5884b7fa, 0x00034802, 0x00000001, 0xfffffffe);
    u32 r2_mod[8] = BIG_INTEGER_CHUNKS8(0x748d9d9, 0x9f59ff11, 0x05d31496, 0x7254398f, 0x2b6cedcb, 0x87925c23, 0xc999e990, 0xf3f29c6d);

    template <u32 TPI>
    struct Params
    {
        static const mont::usize LIMBS = 8;
        typedef cgbn_context_t<TPI>         context_t;
        typedef cgbn_env_t<context_t, LIMBS * 32> env_t;
        static __host__ __device__ __forceinline__ typename env_t::cgbn_t m(int i, int j)
        {
            typename env_t::cgbn_t r;
            for(int k=0; k<j-i; ++k) {
                r._limbs[k] = m[i+k];
            }
            return r;
        }
        // m - 2
        static __host__ __device__ __forceinline__ typename env_t::cgbn_t m_sub2(int i, int j)
        {
            typename env_t::cgbn_t r;
            for(int k=0; k<j-i; ++k) {
                r._limbs[k] = m_sub2[i+k];
            }
            return r;
        }
        // m' = -m^(-1) mod b where b = 2^32
        static const u32 m_prime = 4026531839;
        // r_mod = R mod m,
        static __host__ __device__ __forceinline__ typename env_t::cgbn_t r_mod(int i, int j)
        {
            typename env_t::cgbn_t r;
            for(int k=0; k<j-i; ++k) {
                r._limbs[k] = r_mod[i+k];
            }
            return r;
        }
        // r2_mod = R^2 mod m
        static __host__ __device__ __forceinline__ typename env_t::cgbn_t r2_mod(int i, int j)
        {
            typename env_t::cgbn_t r;
            for(int k=0; k<j-i; ++k) {
                r._limbs[k] = r2_mod[i+k];
            }
            return r;
        }
    };
}