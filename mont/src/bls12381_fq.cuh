#pragma once
#include "field.cuh"

// base field for BLS12-381
namespace bls12381_fq
{
  // bls12381_fq
  // 4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787
  // const auto params = Params {
  //     .m = BIG_INTEGER_CHUNKS12(0x1a0111ea, 0x397fe69a, 0x4b1ba7b6, 0x434bacd7, 0x64774b84, 0xf38512bf, 0x6730d2a0, 0xf6b0f624, 0x1eabfffe, 0xb153ffff, 0xb9feffff, 0xffffaaab),
  //     .mm2 = BIG_INTEGER_CHUNKS12(0x340223d4, 0x72ffcd34, 0x96374f6c, 0x869759ae, 0xc8ee9709, 0xe70a257e, 0xce61a541, 0xed61ec48, 0x3d57fffd, 0x62a7ffff, 0x73fdffff, 0xffff5556),
  //     .r_mod = BIG_INTEGER_CHUNKS12(0x15f65ec3, 0xfa80e493, 0x5c071a97, 0xa256ec6d, 0x77ce5853, 0x70525745, 0x5f489857, 0x53c758ba, 0xebf4000b, 0xc40c0002, 0x76090000, 0x0002fffd),
  //     .r2_mod = BIG_INTEGER_CHUNKS12(0x11988fe5, 0x92cae3aa, 0x9a793e85, 0xb519952d, 0x67eb88a9, 0x939d83c0, 0x8de5476c, 0x4c95b6d5, 0x0a76e6a6, 0x09d104f1, 0xf4df1f34, 0x1c341746),
  //     .m_prime = 4294770685
  // };

  using mont::u32;

  const u32 m_h[12] = BIG_INTEGER_CHUNKS12(0x1a0111ea, 0x397fe69a, 0x4b1ba7b6, 0x434bacd7, 0x64774b84, 0xf38512bf, 0x6730d2a0, 0xf6b0f624, 0x1eabfffe, 0xb153ffff, 0xb9feffff, 0xffffaaab);
  const u32 mm2_h[12] = BIG_INTEGER_CHUNKS12(0x340223d4, 0x72ffcd34, 0x96374f6c, 0x869759ae, 0xc8ee9709, 0xe70a257e, 0xce61a541, 0xed61ec48, 0x3d57fffd, 0x62a7ffff, 0x73fdffff, 0xffff5556);
  const u32 m_sub2_h[12] = BIG_INTEGER_CHUNKS12(0x1a0111ea, 0x397fe69a, 0x4b1ba7b6, 0x434bacd7, 0x64774b84, 0xf38512bf, 0x6730d2a0, 0xf6b0f624, 0x1eabfffe, 0xb153ffff, 0xb9feffff, 4294945449);
  const u32 r_mod_h[12] = BIG_INTEGER_CHUNKS12(0x15f65ec3, 0xfa80e493, 0x5c071a97, 0xa256ec6d, 0x77ce5853, 0x70525745, 0x5f489857, 0x53c758ba, 0xebf4000b, 0xc40c0002, 0x76090000, 0x0002fffd);
  const u32 r2_mod_h[12] = BIG_INTEGER_CHUNKS12(0x11988fe5, 0x92cae3aa, 0x9a793e85, 0xb519952d, 0x67eb88a9, 0x939d83c0, 0x8de5476c, 0x4c95b6d5, 0x0a76e6a6, 0x09d104f1, 0xf4df1f34, 0x1c341746);

  const __device__ u32 m_d[12] = BIG_INTEGER_CHUNKS12(0x1a0111ea, 0x397fe69a, 0x4b1ba7b6, 0x434bacd7, 0x64774b84, 0xf38512bf, 0x6730d2a0, 0xf6b0f624, 0x1eabfffe, 0xb153ffff, 0xb9feffff, 0xffffaaab);
  const __device__ u32 mm2_d[12] = BIG_INTEGER_CHUNKS12(0x340223d4, 0x72ffcd34, 0x96374f6c, 0x869759ae, 0xc8ee9709, 0xe70a257e, 0xce61a541, 0xed61ec48, 0x3d57fffd, 0x62a7ffff, 0x73fdffff, 0xffff5556);
  const __device__ u32 m_sub2_d[12] = BIG_INTEGER_CHUNKS12(0x1a0111ea, 0x397fe69a, 0x4b1ba7b6, 0x434bacd7, 0x64774b84, 0xf38512bf, 0x6730d2a0, 0xf6b0f624, 0x1eabfffe, 0xb153ffff, 0xb9feffff, 4294945449);
  const __device__ u32 r_mod_d[12] = BIG_INTEGER_CHUNKS12(0x15f65ec3, 0xfa80e493, 0x5c071a97, 0xa256ec6d, 0x77ce5853, 0x70525745, 0x5f489857, 0x53c758ba, 0xebf4000b, 0xc40c0002, 0x76090000, 0x0002fffd);
  const __device__ u32 r2_mod_d[12] = BIG_INTEGER_CHUNKS12(0x11988fe5, 0x92cae3aa, 0x9a793e85, 0xb519952d, 0x67eb88a9, 0x939d83c0, 0x8de5476c, 0x4c95b6d5, 0x0a76e6a6, 0x09d104f1, 0xf4df1f34, 0x1c341746);

  struct Params
  {
    static const mont::usize LIMBS = 12;
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
    static const u32 m_prime = 4294770685;
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