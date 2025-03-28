#ifndef BN254_SCALAR_H
#define BN254_SCALAR_H
#include "field.cuh"

#define TPI 2

namespace bn254_scalar
{
  using mont::u32;

  const u32 m_h[8] = BIG_INTEGER_CHUNKS8(0x30644e72, 0xe131a029, 0xb85045b6, 0x8181585d, 0x97816a91, 0x6871ca8d, 0x3c208c16, 0xd87cfd47);
  const u32 mm2_h[8] = BIG_INTEGER_CHUNKS8(0x60c89ce5, 0xc2634053, 0x70a08b6d, 0x0302b0bb, 0x2f02d522, 0xd0e3951a, 0x7841182d, 0xb0f9fa8e);
  const u32 m_sub2_h[8] = BIG_INTEGER_CHUNKS8(0x30644e72, 0xe131a029, 0xb85045b6, 0x8181585d, 0x97816a91, 0x6871ca8d, 0x3c208c16, 0xd87cfd45);
  const u32 r_mod_h[8] = BIG_INTEGER_CHUNKS8(0xe0a77c1, 0x9a07df2f, 0x666ea36f, 0x7879462c, 0x0a78eb28, 0xf5c70b3d, 0xd35d438d, 0xc58f0d9d);
  const u32 r2_mod_h[8] = BIG_INTEGER_CHUNKS8(0x6d89f71, 0xcab8351f, 0x47ab1eff, 0x0a417ff6, 0xb5e71911, 0xd44501fb, 0xf32cfc5b, 0x538afa89);

  const __device__ u32 m_d[8] = BIG_INTEGER_CHUNKS8(0x30644e72, 0xe131a029, 0xb85045b6, 0x8181585d, 0x97816a91, 0x6871ca8d, 0x3c208c16, 0xd87cfd47);
  const __device__ u32 mm2_d[8] = BIG_INTEGER_CHUNKS8(0x60c89ce5, 0xc2634053, 0x70a08b6d, 0x0302b0bb, 0x2f02d522, 0xd0e3951a, 0x7841182d, 0xb0f9fa8e);
  const __device__ u32 m_sub2_d[8] = BIG_INTEGER_CHUNKS8(0x30644e72, 0xe131a029, 0xb85045b6, 0x8181585d, 0x97816a91, 0x6871ca8d, 0x3c208c16, 0xd87cfd45);
  const __device__ u32 r_mod_d[8] = BIG_INTEGER_CHUNKS8(0xe0a77c1, 0x9a07df2f, 0x666ea36f, 0x7879462c, 0x0a78eb28, 0xf5c70b3d, 0xd35d438d, 0xc58f0d9d);
  const __device__ u32 r2_mod_d[8] = BIG_INTEGER_CHUNKS8(0x6d89f71, 0xcab8351f, 0x47ab1eff, 0x0a417ff6, 0xb5e71911, 0xd44501fb, 0xf32cfc5b, 0x538afa89);

//   namespace device_constants
//   {
//     // m = 21888242871839275222246405745257275088696311157297823662689037894645226208583
//     const __device__ Number m = BIG_INTEGER_CHUNKS8(0x30644e72, 0xe131a029, 0xb85045b6, 0x8181585d, 0x97816a91, 0x6871ca8d, 0x3c208c16, 0xd87cfd47);
//     const __device__ Number mm2 = BIG_INTEGER_CHUNKS8(0x60c89ce5, 0xc2634053, 0x70a08b6d, 0x0302b0bb, 0x2f02d522, 0xd0e3951a, 0x7841182d, 0xb0f9fa8e);
//     const __device__ Number m_sub2 = BIG_INTEGER_CHUNKS8(0x30644e72, 0xe131a029, 0xb85045b6, 0x8181585d, 0x97816a91, 0x6871ca8d, 0x3c208c16, 0xd87cfd45);
//     const __device__ Number r_mod = BIG_INTEGER_CHUNKS8(0xe0a77c1, 0x9a07df2f, 0x666ea36f, 0x7879462c, 0x0a78eb28, 0xf5c70b3d, 0xd35d438d, 0xc58f0d9d);
//     const __device__ Number r2_mod = BIG_INTEGER_CHUNKS8(0x6d89f71, 0xcab8351f, 0x47ab1eff, 0x0a417ff6, 0xb5e71911, 0xd44501fb, 0xf32cfc5b, 0x538afa89);
//   }

//   namespace host_constants
//   {
//     const Number m = BIG_INTEGER_CHUNKS8(0x30644e72, 0xe131a029, 0xb85045b6, 0x8181585d, 0x97816a91, 0x6871ca8d, 0x3c208c16, 0xd87cfd47);
//     const Number mm2 = BIG_INTEGER_CHUNKS8(0x60c89ce5, 0xc2634053, 0x70a08b6d, 0x0302b0bb, 0x2f02d522, 0xd0e3951a, 0x7841182d, 0xb0f9fa8e);
//     const Number m_sub2 = BIG_INTEGER_CHUNKS8(0x30644e72, 0xe131a029, 0xb85045b6, 0x8181585d, 0x97816a91, 0x6871ca8d, 0x3c208c16, 0xd87cfd45);
//     const Number r_mod = BIG_INTEGER_CHUNKS8(0xe0a77c1, 0x9a07df2f, 0x666ea36f, 0x7879462c, 0x0a78eb28, 0xf5c70b3d, 0xd35d438d, 0xc58f0d9d);
//     const Number r2_mod = BIG_INTEGER_CHUNKS8(0x6d89f71, 0xcab8351f, 0x47ab1eff, 0x0a417ff6, 0xb5e71911, 0xd44501fb, 0xf32cfc5b, 0x538afa89);
//   }

  struct Params
  {
    static const mont::usize LIMBS = 8;
    template <class cgbn_env_t>
    static __host__ __device__ __forceinline__ typename cgbn_env_t::cgbn_t m(const cgbn_env_t &env, int i, int j)
    {
      assert((j-i) == (LIMBS/TPI));
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
        assert((j-i) == (LIMBS/TPI));
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
      assert((j-i) == (LIMBS/TPI));
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
    static const u32 m_prime = 3834012553;
    // r_mod = R mod m,
    template <class cgbn_env_t>
    static __host__ __device__ __forceinline__ typename cgbn_env_t::cgbn_t r_mod(const cgbn_env_t &env, int i, int j)
    {
      assert((j-i) == (LIMBS/TPI));
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
      assert((j-i) == (LIMBS/TPI));
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

#endif
