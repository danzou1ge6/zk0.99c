#pragma once
#include "field.cuh"

namespace mnt4753_fq
{
      // 41898490967918953402344214791240637128170709919953949071783502921025352812571106773058893763790338921418070971888253786114353726529584385201591605722013126468931404347949840543007986327743462853720628051692141265303114721689601
      // const auto params = Params {
      //     .m = BIG_INTEGER_CHUNKS8(0x1c4c6, 0x2d92c411, 0x10229022, 0xeee2cdad, 0xb7f99750, 0x5b8fafed, 0x5eb7e8f9, 0x6c97d873, 0x07fdb925, 0xe8a0ed8d, 0x99d124d9, 0xa15af79d, 0xb117e776, 0xf218059d, 0xb80f0da5, 0xcb537e38, 0x685acce9, 0x767254a4, 0x63881071, 0x9ac425f0, 0xe39d5452, 0x2cdd119f, 0x5e9063de, 0x245e8001),
      //     .mm2 = BIG_INTEGER_CHUNKS8(0x3898c, 0x5b258822, 0x20452045, 0xddc59b5b, 0x6ff32ea0, 0xb71f5fda, 0xbd6fd1f2, 0xd92fb0e6, 0x0ffb724b, 0xd141db1b, 0x33a249b3, 0x42b5ef3b, 0x622fceed, 0xe4300b3b, 0x701e1b4b, 0x96a6fc70, 0xd0b599d2, 0xece4a948, 0xc71020e3, 0x35884be1, 0xc73aa8a4, 0x59ba233e, 0xbd20c7bc, 0x48bd0002),
      //     .r_mod = BIG_INTEGER_CHUNKS8(0x7b47, 0x9ec8e242, 0x95455fb3, 0x1ff9a195, 0x0fa47edb, 0x3865e88c, 0x4074c9cb, 0xfd8ca621, 0x598b4302, 0xd2f00a62, 0x320c3bb7, 0x13338559, 0x1e0f4d8a, 0xcf031d68, 0xed269c94, 0x2108976f, 0x79589819, 0xc788b601, 0x97c3e4a0, 0xcd14572e, 0x91cd31c6, 0x5a034686, 0x98a8ecab, 0xd9dc6f42),
      //     .r2_mod = BIG_INTEGER_CHUNKS8(0x2a33, 0xe89cb485, 0xb081f15b, 0xcbfdacaf, 0x8e460575, 0x4c381723, 0x2505daf1, 0xf4a81245, 0xe03c79ca, 0xc4f7ef07, 0xa8c86d46, 0x04a3b597, 0x2f47839e, 0xf88d7ce8, 0x80a46659, 0xff6f3ddf, 0xa896a656, 0xa0714c7d, 0xa24bea56, 0x242b3507, 0xc7d9ff8e, 0x7df03c0a, 0x84717088, 0xcfd190c8),
      //     .m_prime = 3831398399
      // };
    
      using mont::u32;

      const u32 m_h[24] = BIG_INTEGER_CHUNKS24(0x1c4c6, 0x2d92c411, 0x10229022, 0xeee2cdad, 0xb7f99750, 0x5b8fafed, 0x5eb7e8f9, 0x6c97d873, 0x07fdb925, 0xe8a0ed8d, 0x99d124d9, 0xa15af79d, 0xb117e776, 0xf218059d, 0xb80f0da5, 0xcb537e38, 0x685acce9, 0x767254a4, 0x63881071, 0x9ac425f0, 0xe39d5452, 0x2cdd119f, 0x5e9063de, 0x245e8001);
      const u32 mm2_h[24] = BIG_INTEGER_CHUNKS24(0x3898c, 0x5b258822, 0x20452045, 0xddc59b5b, 0x6ff32ea0, 0xb71f5fda, 0xbd6fd1f2, 0xd92fb0e6, 0x0ffb724b, 0xd141db1b, 0x33a249b3, 0x42b5ef3b, 0x622fceed, 0xe4300b3b, 0x701e1b4b, 0x96a6fc70, 0xd0b599d2, 0xece4a948, 0xc71020e3, 0x35884be1, 0xc73aa8a4, 0x59ba233e, 0xbd20c7bc, 0x48bd0002);
      const u32 m_sub2_h[24] = BIG_INTEGER_CHUNKS24(0x1c4c6, 0x2d92c411, 0x10229022, 0xeee2cdad, 0xb7f99750, 0x5b8fafed, 0x5eb7e8f9, 0x6c97d873, 0x07fdb925, 0xe8a0ed8d, 0x99d124d9, 0xa15af79d, 0xb117e776, 0xf218059d, 0xb80f0da5, 0xcb537e38, 0x685acce9, 0x767254a4, 0x63881071, 0x9ac425f0, 0xe39d5452, 0x2cdd119f, 0x5e9063de, 610172927);
      const u32 r_mod_h[24] = BIG_INTEGER_CHUNKS24(0x7b47, 0x9ec8e242, 0x95455fb3, 0x1ff9a195, 0x0fa47edb, 0x3865e88c, 0x4074c9cb, 0xfd8ca621, 0x598b4302, 0xd2f00a62, 0x320c3bb7, 0x13338559, 0x1e0f4d8a, 0xcf031d68, 0xed269c94, 0x2108976f, 0x79589819, 0xc788b601, 0x97c3e4a0, 0xcd14572e, 0x91cd31c6, 0x5a034686, 0x98a8ecab, 0xd9dc6f42);
      const u32 r2_mod_h[24] = BIG_INTEGER_CHUNKS24(0x2a33, 0xe89cb485, 0xb081f15b, 0xcbfdacaf, 0x8e460575, 0x4c381723, 0x2505daf1, 0xf4a81245, 0xe03c79ca, 0xc4f7ef07, 0xa8c86d46, 0x04a3b597, 0x2f47839e, 0xf88d7ce8, 0x80a46659, 0xff6f3ddf, 0xa896a656, 0xa0714c7d, 0xa24bea56, 0x242b3507, 0xc7d9ff8e, 0x7df03c0a, 0x84717088, 0xcfd190c8);
      
      const __device__ u32 m_d[24] = BIG_INTEGER_CHUNKS24(0x1c4c6, 0x2d92c411, 0x10229022, 0xeee2cdad, 0xb7f99750, 0x5b8fafed, 0x5eb7e8f9, 0x6c97d873, 0x07fdb925, 0xe8a0ed8d, 0x99d124d9, 0xa15af79d, 0xb117e776, 0xf218059d, 0xb80f0da5, 0xcb537e38, 0x685acce9, 0x767254a4, 0x63881071, 0x9ac425f0, 0xe39d5452, 0x2cdd119f, 0x5e9063de, 0x245e8001);
      const __device__ u32 mm2_d[24] = BIG_INTEGER_CHUNKS24(0x3898c, 0x5b258822, 0x20452045, 0xddc59b5b, 0x6ff32ea0, 0xb71f5fda, 0xbd6fd1f2, 0xd92fb0e6, 0x0ffb724b, 0xd141db1b, 0x33a249b3, 0x42b5ef3b, 0x622fceed, 0xe4300b3b, 0x701e1b4b, 0x96a6fc70, 0xd0b599d2, 0xece4a948, 0xc71020e3, 0x35884be1, 0xc73aa8a4, 0x59ba233e, 0xbd20c7bc, 0x48bd0002);
      const __device__ u32 m_sub2_d[24] = BIG_INTEGER_CHUNKS24(0x1c4c6, 0x2d92c411, 0x10229022, 0xeee2cdad, 0xb7f99750, 0x5b8fafed, 0x5eb7e8f9, 0x6c97d873, 0x07fdb925, 0xe8a0ed8d, 0x99d124d9, 0xa15af79d, 0xb117e776, 0xf218059d, 0xb80f0da5, 0xcb537e38, 0x685acce9, 0x767254a4, 0x63881071, 0x9ac425f0, 0xe39d5452, 0x2cdd119f, 0x5e9063de, 610172927);
      const __device__ u32 r_mod_d[24] = BIG_INTEGER_CHUNKS24(0x7b47, 0x9ec8e242, 0x95455fb3, 0x1ff9a195, 0x0fa47edb, 0x3865e88c, 0x4074c9cb, 0xfd8ca621, 0x598b4302, 0xd2f00a62, 0x320c3bb7, 0x13338559, 0x1e0f4d8a, 0xcf031d68, 0xed269c94, 0x2108976f, 0x79589819, 0xc788b601, 0x97c3e4a0, 0xcd14572e, 0x91cd31c6, 0x5a034686, 0x98a8ecab, 0xd9dc6f42);
      const __device__ u32 r2_mod_d[24] = BIG_INTEGER_CHUNKS24(0x2a33, 0xe89cb485, 0xb081f15b, 0xcbfdacaf, 0x8e460575, 0x4c381723, 0x2505daf1, 0xf4a81245, 0xe03c79ca, 0xc4f7ef07, 0xa8c86d46, 0x04a3b597, 0x2f47839e, 0xf88d7ce8, 0x80a46659, 0xff6f3ddf, 0xa896a656, 0xa0714c7d, 0xa24bea56, 0x242b3507, 0xc7d9ff8e, 0x7df03c0a, 0x84717088, 0xcfd190c8);
    
      struct Params
      {
        static const mont::usize LIMBS = 24;
        template <class cgbn_env_t>
        static __host__ __device__ __forceinline__ typename cgbn_env_t::cgbn_t m(const cgbn_env_t &env, int i, int j)
        {
          typename cgbn_env_t::cgbn_t r;
          for(int k=0; k<j-i; ++k) {
    #ifdef __CUDA_ARCH__
                r._limbs[k] = m_d[i+k];
    #else
                r._limbs[k] = m_h[i+k];
    #endif
          }
          return r;
        }
        static __host__ __device__ __forceinline__ const u32* m_all()
        {
    #ifdef __CUDA_ARCH__
            return &m_d[0];
    #else
            return &m_h[0];
    #endif
        }
          // m * 2
          template <class cgbn_env_t>
          static __host__ __device__ __forceinline__ typename cgbn_env_t::cgbn_t mm2(const cgbn_env_t &env, int i, int j)
          {
            typename cgbn_env_t::cgbn_t r;
            for(int k=0; k<j-i; ++k) {
    #ifdef __CUDA_ARCH__
              r._limbs[k] = mm2_d[i+k];
    #else
              r._limbs[k] = mm2_h[i+k];
    #endif
            }
            return r;
          }
          static __host__ __device__ __forceinline__ const u32* mm2_all()
        {
    #ifdef __CUDA_ARCH__
            return &mm2_d[0];
    #else
            return &mm2_h[0];
    #endif
        }
        // m - 2
        template <class cgbn_env_t>
        static __host__ __device__ __forceinline__ typename cgbn_env_t::cgbn_t m_sub2(const cgbn_env_t &env, int i, int j)
        {
          typename cgbn_env_t::cgbn_t r;
          for(int k=0; k<j-i; ++k) {
    #ifdef __CUDA_ARCH__
            r._limbs[k] = m_sub2_d[i+k];
    #else
            r._limbs[k] = m_sub2_h[i+k];
    #endif
          }
          return r;
        }
        static __host__ __device__ __forceinline__ const u32* m_sub2_all()
        {
    #ifdef __CUDA_ARCH__
          return &m_sub2_d[0];
    #else
          return &m_sub2_h[0];
    #endif
        }
        // m' = -m^(-1) mod b where b = 2^32
        static const u32 m_prime = 3831398399;
        // r_mod = R mod m,
        template <class cgbn_env_t>
        static __host__ __device__ __forceinline__ typename cgbn_env_t::cgbn_t r_mod(const cgbn_env_t &env, int i, int j)
        {
          typename cgbn_env_t::cgbn_t r;
          for(int k=0; k<j-i; ++k) {
    #ifdef __CUDA_ARCH__
            r._limbs[k] = r_mod_d[i+k];
    #else
            r._limbs[k] = r_mod_h[i+k];
    #endif
          }
          return r;
        }
        static __host__ __device__ __forceinline__ const u32 r_mod_single(int i)
        {
    #ifdef __CUDA_ARCH__
          return r_mod_d[i];
    #else
          return r_mod_h[i];
    #endif
        }
        // r2_mod = R^2 mod m
        template <class cgbn_env_t>
        static __host__ __device__ __forceinline__ typename cgbn_env_t::cgbn_t r2_mod(const cgbn_env_t &env, int i, int j)
        {
          typename cgbn_env_t::cgbn_t r;
          for(int k=0; k<j-i; ++k) {
    #ifdef __CUDA_ARCH__
            r._limbs[k] = r2_mod_d[i+k];
    #else
            r._limbs[k] = r2_mod_h[i+k];
    #endif
          }
          return r;
        }
        static __host__ __device__ __forceinline__ const u32* r2_mod_all()
        {
    #ifdef __CUDA_ARCH__
          return &r2_mod_d[0];
    #else
          return &r2_mod_h[0];
    #endif
        }
      };
    }