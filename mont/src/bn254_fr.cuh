#pragma once
#include "field.cuh"

#define TPI 2

namespace bn254_fr
{
    // bn254_fr
    // 21888242871839275222246405745257275088548364400416034343698204186575808495617
    // const auto params_bn254_fr = mont256::Params {
    //   .m = BIG_INTEGER_CHUNKS8(0x30644e72, 0xe131a029, 0xb85045b6, 0x8181585d, 0x2833e848, 0x79b97091, 0x43e1f593, 0xf0000001),
    //   .r_mod = BIG_INTEGER_CHUNKS8(0xe0a77c1, 0x9a07df2f, 0x666ea36f, 0x7879462e, 0x36fc7695, 0x9f60cd29, 0xac96341c, 0x4ffffffb),
    //   .r2_mod = BIG_INTEGER_CHUNKS8(0x216d0b1, 0x7f4e44a5, 0x8c49833d, 0x53bb8085, 0x53fe3ab1, 0xe35c59e3, 0x1bb8e645, 0xae216da7),
    //   .m_prime = 4026531839     
    // };

    // using Number = mont::Number<8>;
    using mont::u32;
    u32 m[8] = BIG_INTEGER_CHUNKS8(0x30644e72, 0xe131a029, 0xb85045b6, 0x8181585d, 0x2833e848, 0x79b97091, 0x43e1f593, 0xf0000001);
    u32 mm2[8] = BIG_INTEGER_CHUNKS8(0x60c89ce5, 0xc2634053, 0x70a08b6d, 0x0302b0ba, 0x5067d090, 0xf372e122, 0x87c3eb27, 0xe0000002);
    u32 m_sub2[8] = BIG_INTEGER_CHUNKS8(0x30644e72, 0xe131a029, 0xb85045b6, 0x8181585d, 0x2833e848, 0x79b97091, 0x43e1f593, 0xefffffff);
    u32 r_mod[8] = BIG_INTEGER_CHUNKS8(0xe0a77c1, 0x9a07df2f, 0x666ea36f, 0x7879462e, 0x36fc7695, 0x9f60cd29, 0xac96341c, 0x4ffffffb);
    u32 r2_mod[8] = BIG_INTEGER_CHUNKS8(0x216d0b1, 0x7f4e44a5, 0x8c49833d, 0x53bb8085, 0x53fe3ab1, 0xe35c59e3, 0x1bb8e645, 0xae216da7);

    // namespace device_constants
    // {
    //     // m = 21888242871839275222246405745257275088548364400416034343698204186575808495617
    //     const __device__ Number m = BIG_INTEGER_CHUNKS8(0x30644e72, 0xe131a029, 0xb85045b6, 0x8181585d, 0x2833e848, 0x79b97091, 0x43e1f593, 0xf0000001);
    //     const __device__ Number mm2 = BIG_INTEGER_CHUNKS8(0x60c89ce5, 0xc2634053, 0x70a08b6d, 0x0302b0ba, 0x5067d090, 0xf372e122, 0x87c3eb27, 0xe0000002);
    //     const __device__ Number m_sub2 = BIG_INTEGER_CHUNKS8(0x30644e72, 0xe131a029, 0xb85045b6, 0x8181585d, 0x2833e848, 0x79b97091, 0x43e1f593, 0xefffffff);
    //     const __device__ Number r_mod = BIG_INTEGER_CHUNKS8(0xe0a77c1, 0x9a07df2f, 0x666ea36f, 0x7879462e, 0x36fc7695, 0x9f60cd29, 0xac96341c, 0x4ffffffb);
    //     const __device__ Number r2_mod = BIG_INTEGER_CHUNKS8(0x216d0b1, 0x7f4e44a5, 0x8c49833d, 0x53bb8085, 0x53fe3ab1, 0xe35c59e3, 0x1bb8e645, 0xae216da7);
    // }

    // namespace host_constants
    // {
    //     const Number m = BIG_INTEGER_CHUNKS8(0x30644e72, 0xe131a029, 0xb85045b6, 0x8181585d, 0x2833e848, 0x79b97091, 0x43e1f593, 0xf0000001);
    //     const Number mm2 = BIG_INTEGER_CHUNKS8(0x60c89ce5, 0xc2634053, 0x70a08b6d, 0x0302b0ba, 0x5067d090, 0xf372e122, 0x87c3eb27, 0xe0000002);
    //     const Number m_sub2 = BIG_INTEGER_CHUNKS8(0x30644e72, 0xe131a029, 0xb85045b6, 0x8181585d, 0x2833e848, 0x79b97091, 0x43e1f593, 0xefffffff);
    //     const Number r_mod = BIG_INTEGER_CHUNKS8(0xe0a77c1, 0x9a07df2f, 0x666ea36f, 0x7879462e, 0x36fc7695, 0x9f60cd29, 0xac96341c, 0x4ffffffb);
    //     const Number r2_mod = BIG_INTEGER_CHUNKS8(0x216d0b1, 0x7f4e44a5, 0x8c49833d, 0x53bb8085, 0x53fe3ab1, 0xe35c59e3, 0x1bb8e645, 0xae216da7);
    // }

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
        // m * 2
        static __host__ __device__ __forceinline__ typename env_t::cgbn_t mm2(int i, int j)
        {
            typename env_t::cgbn_t r;
            for(int k=0; k<j-i; ++k) {
                  r._limbs[k] = mm2[i+k];
            }
            return r;
        }
        // m - 2
        static __host__ __device__ __forceinline__ typename env_t::cgbn_t m_sub2()
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
        static __host__ __device__ __forceinline__ typename env_t::cgbn_t r_mod()
        {
            typename env_t::cgbn_t r;
            for(int k=0; k<j-i; ++k) {
                r._limbs[k] = r_mod[i+k];
            }
            return r;
        }
        // r2_mod = R^2 mod m
        static __host__ __device__ __forceinline__ typename env_t::cgbn_t r2_mod()
        {
            typename env_t::cgbn_t r;
            for(int k=0; k<j-i; ++k) {
                r._limbs[k] = r2_mod[i+k];
            }
            return r;
        }
    };
}