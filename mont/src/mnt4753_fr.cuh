#pragma once
#include "field.cuh"

#define TPI 2

namespace mnt4753_fr
{
    // .m = BIG_INTEGER_CHUNKS8(0x1c4c6, 0x2d92c411, 0x10229022, 0xeee2cdad, 0xb7f99750, 0x5b8fafed, 0x5eb7e8f9, 0x6c97d873, 0x07fdb925, 0xe8a0ed8d, 0x99d124d9, 0xa15af79d, 0xb26c5c28, 0xc859a99b, 0x3eebca94, 0x29212636, 0xb9dff976, 0x34993aa4, 0xd6c381bc, 0x3f005797, 0x4ea09917, 0x0fa13a4f, 0xd90776e2, 0x40000001),
    // .mm2 = BIG_INTEGER_CHUNKS8(0x3898c, 0x5b258822, 0x20452045, 0xddc59b5b, 0x6ff32ea0, 0xb71f5fda, 0xbd6fd1f2, 0xd92fb0e6, 0x0ffb724b, 0xd141db1b, 0x33a249b3, 0x42b5ef3b, 0x64d8b851, 0x90b35336, 0x7dd79528, 0x52424c6d, 0x73bff2ec, 0x69327549, 0xad870378, 0x7e00af2e, 0x9d41322e, 0x1f42749f, 0xb20eedc4, 0x80000002),
    // .r_mod = BIG_INTEGER_CHUNKS8(0x7b47, 0x9ec8e242, 0x95455fb3, 0x1ff9a195, 0x0fa47edb, 0x3865e88c, 0x4074c9cb, 0xfd8ca621, 0x598b4302, 0xd2f00a62, 0x320c3bb7, 0x13338498, 0x9fbca908, 0xde0ccb62, 0xab0c4ee6, 0xd3e6dad4, 0x0f725cae, 0xc549c0da, 0xa1ebd2d9, 0x0c79e179, 0x4eb16817, 0xb589cea8, 0xb9968014, 0x7fff6f42),
    // .r2_mod = BIG_INTEGER_CHUNKS8(0x5b5, 0x8037e0e4, 0xd9e817a8, 0xfb44b3c9, 0xacc27988, 0xf3d9a316, 0xc8a0ff01, 0x493bdcef, 0x99be80f2, 0xee12ee8e, 0xded52121, 0xecec77cf, 0xc7285529, 0xbe54a3f4, 0x7955876c, 0xc35ee94e, 0x6bd8c6c6, 0xc49edc38, 0xcdbe6702, 0x009569cb, 0x70a50fa9, 0xee48d127, 0x3f9c69c7, 0xb7f4c8d1),
    // .m_prime = 1073741823

    using mont::u32;
    u32 m[24] = BIG_INTEGER_CHUNKS24(0x1c4c6, 0x2d92c411, 0x10229022, 0xeee2cdad, 0xb7f99750, 0x5b8fafed, 0x5eb7e8f9, 0x6c97d873, 0x07fdb925, 0xe8a0ed8d, 0x99d124d9, 0xa15af79d, 0xb26c5c28, 0xc859a99b, 0x3eebca94, 0x29212636, 0xb9dff976, 0x34993aa4, 0xd6c381bc, 0x3f005797, 0x4ea09917, 0x0fa13a4f, 0xd90776e2, 0x40000001);
    u32 m_sub2[24] = BIG_INTEGER_CHUNKS24(0x1c4c6, 0x2d92c411, 0x10229022, 0xeee2cdad, 0xb7f99750, 0x5b8fafed, 0x5eb7e8f9, 0x6c97d873, 0x07fdb925, 0xe8a0ed8d, 0x99d124d9, 0xa15af79d, 0xb26c5c28, 0xc859a99b, 0x3eebca94, 0x29212636, 0xb9dff976, 0x34993aa4, 0xd6c381bc, 0x3f005797, 0x4ea09917, 0x0fa13a4f, 0xd90776e2, 1073741823);
    u32 r_mod[24] = BIG_INTEGER_CHUNKS24(0x7b47, 0x9ec8e242, 0x95455fb3, 0x1ff9a195, 0x0fa47edb, 0x3865e88c, 0x4074c9cb, 0xfd8ca621, 0x598b4302, 0xd2f00a62, 0x320c3bb7, 0x13338498, 0x9fbca908, 0xde0ccb62, 0xab0c4ee6, 0xd3e6dad4, 0x0f725cae, 0xc549c0da, 0xa1ebd2d9, 0x0c79e179, 0x4eb16817, 0xb589cea8, 0xb9968014, 0x7fff6f42);
    u32 r2_mod[24] = BIG_INTEGER_CHUNKS24(0x5b5, 0x8037e0e4, 0xd9e817a8, 0xfb44b3c9, 0xacc27988, 0xf3d9a316, 0xc8a0ff01, 0x493bdcef, 0x99be80f2, 0xee12ee8e, 0xded52121, 0xecec77cf, 0xc7285529, 0xbe54a3f4, 0x7955876c, 0xc35ee94e, 0x6bd8c6c6, 0xc49edc38, 0xcdbe6702, 0x009569cb, 0x70a50fa9, 0xee48d127, 0x3f9c69c7, 0xb7f4c8d1);

    struct Params
    {
        static const mont::usize LIMBS = 24;
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