#include "field.cuh"

namespace bn254_scalar
{
  using mont::Number;
  using mont::u32;

  namespace device_constants
  {
    // m = 21888242871839275222246405745257275088696311157297823662689037894645226208583
    const __device__ Number<8> m = BIG_INTEGER_CHUNKS8(0x30644e72, 0xe131a029, 0xb85045b6, 0x8181585d, 0x97816a91, 0x6871ca8d, 0x3c208c16, 0xd87cfd47);
    const __device__ Number<8> m_sub2 = BIG_INTEGER_CHUNKS8(0x30644e72, 0xe131a029, 0xb85045b6, 0x8181585d, 0x97816a91, 0x6871ca8d, 0x3c208c16, 0xd87cfd45);
    const __device__ Number<8> r_mod = BIG_INTEGER_CHUNKS8(0xe0a77c1, 0x9a07df2f, 0x666ea36f, 0x7879462c, 0x0a78eb28, 0xf5c70b3d, 0xd35d438d, 0xc58f0d9d);
    const __device__ Number<8> r2_mod = BIG_INTEGER_CHUNKS8(0x6d89f71, 0xcab8351f, 0x47ab1eff, 0x0a417ff6, 0xb5e71911, 0xd44501fb, 0xf32cfc5b, 0x538afa89);
  }

  namespace host_constants
  {
    const Number<8> m = BIG_INTEGER_CHUNKS8(0x30644e72, 0xe131a029, 0xb85045b6, 0x8181585d, 0x97816a91, 0x6871ca8d, 0x3c208c16, 0xd87cfd47);
    const Number<8> m_sub2 = BIG_INTEGER_CHUNKS8(0x30644e72, 0xe131a029, 0xb85045b6, 0x8181585d, 0x97816a91, 0x6871ca8d, 0x3c208c16, 0xd87cfd45);
    const Number<8> r_mod = BIG_INTEGER_CHUNKS8(0xe0a77c1, 0x9a07df2f, 0x666ea36f, 0x7879462c, 0x0a78eb28, 0xf5c70b3d, 0xd35d438d, 0xc58f0d9d);
    const Number<8> r2_mod = BIG_INTEGER_CHUNKS8(0x6d89f71, 0xcab8351f, 0x47ab1eff, 0x0a417ff6, 0xb5e71911, 0xd44501fb, 0xf32cfc5b, 0x538afa89);
  }

  struct Params
  {
    static const mont::usize LIMBS = 8;
    static const __host__ __device__ __forceinline__ Number<8> m()
    {
#ifdef __CUDA_ARCH__
      return device_constants::m;
#else
      return host_constants::m;
#endif
    }
    // m - 2
    static const __host__ __device__ __forceinline__ Number<8> m_sub2()
    {
#ifdef __CUDA_ARCH__
      return device_constants::m_sub2;
#else
      return host_constants::m_sub2;
#endif
    }
    // m' = -m^(-1) mod b where b = 2^32
    static const u32 m_prime = 3834012553;
    // r_mod = R mod m,
    static const __host__ __device__ __forceinline__ Number<8> r_mod()
    {
#ifdef __CUDA_ARCH__
      return device_constants::r_mod;
#else
      return host_constants::r_mod;
#endif
    }
    // r2_mod = R^2 mod m
    static const __host__ __device__ __forceinline__ Number<8> r2_mod()
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
