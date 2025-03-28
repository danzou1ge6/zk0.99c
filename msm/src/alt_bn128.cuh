#ifndef ALT_BN128_H
#define ALT_BN128_H

#include "../../mont/src/alt_bn128_fq.cuh"
#include "curve_xyzz.cuh"

namespace alt_bn128_g1
{
  using alt_bn128_fq::Element;
  using Number = mont::Number<Element::LIMBS>;

  namespace device_constants
  {
    constexpr __device__ Element b = Element(Number(BIG_INTEGER_CHUNKS8(
        0x2a1f6744, 0xce179d8e, 0x334bea4e, 0x696bd28a, 0xa4f563c0, 0xde22677d, 0x05c29c54, 0xeffffff1)));
    constexpr __device__ Element b3 = Element(Number(BIG_INTEGER_CHUNKS8(
        0x1d9598e8, 0xa7e39857, 0x2943337e, 0x3940c6e5, 0x9e785ab1, 0xa6f45554, 0x8983e9d6, 0xefffffd1)));
    constexpr __device__ Element a = Element(Number(BIG_INTEGER_CHUNKS8(
        0, 0, 0, 0, 0, 0, 0, 0)));
  }

  namespace host_constants
  {
    constexpr Element b = Element(Number(BIG_INTEGER_CHUNKS8(
        0x2a1f6744, 0xce179d8e, 0x334bea4e, 0x696bd28a, 0xa4f563c0, 0xde22677d, 0x05c29c54, 0xeffffff1)));
    constexpr Element b3 = Element(Number(BIG_INTEGER_CHUNKS8(
        0x1d9598e8, 0xa7e39857, 0x2943337e, 0x3940c6e5, 0x9e785ab1, 0xa6f45554, 0x8983e9d6, 0xefffffd1)));
    constexpr Element a = Element(Number(BIG_INTEGER_CHUNKS8(
        0, 0, 0, 0, 0, 0, 0, 0)));
  }

  struct Params
  {

    static constexpr __device__ __host__ __forceinline__
        Element
        a()
    {
#ifdef __CUDA_ARCH__
      return device_constants::a;
#else
      return host_constants::a;
#endif
    }

    static constexpr __device__ __host__ __forceinline__
        Element
        b()
    {
#ifdef __CUDA_ARCH__
      return device_constants::b;
#else
      return host_constants::b;
#endif
    }

    static __device__ __host__ __forceinline__
        Element
        b3()
    {
#ifdef __CUDA_ARCH__
      return device_constants::b3;
#else
      return host_constants::b3;
#endif
    }

    static constexpr __device__ __host__ __forceinline__ bool
    allow_lazy_modulo()
    {
      return false;
    }
  };

  using Point = curve::EC<Params, Element>::PointXYZZ;
  using PointAffine = curve::EC<Params, Element>::PointAffine;
}

namespace alt_bn128_g2
{
  using alt_bn128_fq2::Element;
  using Number = mont::Number<Element::OnceType::LIMBS>;

  // a = 0, b = 3 (x + 9)^(-1)
  namespace device_constants
  {
    constexpr __device__ Element b = Element(Number(BIG_INTEGER_CHUNKS8(
                                                 0x1633c28a, 0x35bc5b45, 0x42622236, 0x8866ed48, 0xf53e8a72, 0x128b76c0, 0x1026d01c, 0xab20da56)),
                                             Number(BIG_INTEGER_CHUNKS8(
                                                 0x2608e637, 0x73c59305, 0xb2118a2d, 0x7dbd035c, 0x1fbce7d8, 0x69e2e11f, 0x0c5f8802, 0xbf29ec5f)));
    constexpr __device__ Element b3 = Element(Number(BIG_INTEGER_CHUNKS8(
                                                  0x1236f92b, 0xc00371a6, 0x0ed620ed, 0x17b36f7d, 0xb787b70d, 0xbde8f3ae, 0xec927ac2, 0x11628f01)),
                                              Number(BIG_INTEGER_CHUNKS8(
                                                  0x115215c0, 0x98ed78bd, 0xa594131b, 0x7634595a, 0x0ecee6f8, 0x4a35c23a, 0x9d5aace0, 0x5d7dc51b)));
    constexpr __device__ Element a = Element(Number(BIG_INTEGER_CHUNKS8(
                                                 0, 0, 0, 0, 0, 0, 0, 0)),
                                             Number(BIG_INTEGER_CHUNKS8(0, 0, 0, 0, 0, 0, 0, 0)));
  }

  namespace host_constants
  {
    constexpr Element b = Element(Number(BIG_INTEGER_CHUNKS8(
                                      0x1633c28a, 0x35bc5b45, 0x42622236, 0x8866ed48, 0xf53e8a72, 0x128b76c0, 0x1026d01c, 0xab20da56)),
                                  Number(BIG_INTEGER_CHUNKS8(
                                      0x2608e637, 0x73c59305, 0xb2118a2d, 0x7dbd035c, 0x1fbce7d8, 0x69e2e11f, 0x0c5f8802, 0xbf29ec5f)));
    constexpr Element b3 = Element(Number(BIG_INTEGER_CHUNKS8(
                                       0x1236f92b, 0xc00371a6, 0x0ed620ed, 0x17b36f7d, 0xb787b70d, 0xbde8f3ae, 0xec927ac2, 0x11628f01)),
                                   Number(BIG_INTEGER_CHUNKS8(
                                       0x115215c0, 0x98ed78bd, 0xa594131b, 0x7634595a, 0x0ecee6f8, 0x4a35c23a, 0x9d5aace0, 0x5d7dc51b)));
    constexpr Element a = Element(Number(BIG_INTEGER_CHUNKS8(
                                      0, 0, 0, 0, 0, 0, 0, 0)),
                                  Number(BIG_INTEGER_CHUNKS8(0, 0, 0, 0, 0, 0, 0, 0)));
  }

  struct Params
  {

    static constexpr __device__ __host__ __forceinline__
        Element
        a()
    {
#ifdef __CUDA_ARCH__
      return device_constants::a;
#else
      return host_constants::a;
#endif
    }

    static constexpr __device__ __host__ __forceinline__
        Element
        b()
    {
#ifdef __CUDA_ARCH__
      return device_constants::b;
#else
      return host_constants::b;
#endif
    }

    static __device__ __host__ __forceinline__
        Element
        b3()
    {
#ifdef __CUDA_ARCH__
      return device_constants::b3;
#else
      return host_constants::b3;
#endif
    }

    static constexpr __device__ __host__ __forceinline__ bool
    allow_lazy_modulo()
    {
      return true;
    }
  };

  using Point = curve::EC<Params, Element>::PointXYZZ;
  using PointAffine = curve::EC<Params, Element>::PointAffine;
}

#endif