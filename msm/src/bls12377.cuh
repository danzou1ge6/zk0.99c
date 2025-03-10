#ifndef BN254_H
#define BN254_H

#include "../../mont/src/bls12377_fq.cuh"
#include "../../mont/src/bls12377_fr.cuh"
#include "curve_xyzz.cuh"

namespace bls12377
{
  using Element = bls12377_fq::Element; // base field for curves
  using BaseNumber = mont::Number<Element::LIMBS>; // number for base field
  using Field = bls12377_fr::Element; // field for scalars
  using Number = mont::Number<Field::LIMBS>; // number for scalar field

  namespace device_constants
  {
    constexpr __device__ Element b = Element(BaseNumber(BIG_INTEGER_CHUNKS12(0x8d6661, 0xe2fdf49a, 0x4cf495bf, 0x803c84e8, 0x7b4e97b7, 0x6e7c6305, 0x9f7db3a9, 0x8a7d3ff2, 0x51409f83, 0x7fffffb1, 0x02cdffff, 0xffffff68)));
    constexpr __device__ Element b3 = Element(BaseNumber(BIG_INTEGER_CHUNKS12(0x1a83325, 0xa8f9ddce, 0xe6ddc13e, 0x80b58eb9, 0x71ebc726, 0x4b752910, 0xde791afc, 0x9f77bfd6, 0xf3c1de8a, 0x7fffff13, 0x0869ffff, 0xfffffe38)));
    constexpr __device__ Element a = Element(BaseNumber(BIG_INTEGER_CHUNKS12(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
  }

  namespace host_constants
  {
    constexpr Element b = Element(BaseNumber(BIG_INTEGER_CHUNKS12(0x8d6661, 0xe2fdf49a, 0x4cf495bf, 0x803c84e8, 0x7b4e97b7, 0x6e7c6305, 0x9f7db3a9, 0x8a7d3ff2, 0x51409f83, 0x7fffffb1, 0x02cdffff, 0xffffff68)));
    constexpr Element b3 = Element(BaseNumber(BIG_INTEGER_CHUNKS12(0x1a83325, 0xa8f9ddce, 0xe6ddc13e, 0x80b58eb9, 0x71ebc726, 0x4b752910, 0xde791afc, 0x9f77bfd6, 0xf3c1de8a, 0x7fffff13, 0x0869ffff, 0xfffffe38)));
    constexpr Element a = Element(BaseNumber(BIG_INTEGER_CHUNKS12(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
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

    static constexpr __device__ __host__ __forceinline__
        bool
        allow_lazy_modulo()
    {
        return true;
    }
  };

  using Point = curve::EC<Params, Element>::PointXYZZ;
  using PointAffine = curve::EC<Params, Element>::PointAffine;
}

#endif