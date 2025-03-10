#ifndef BN254_H
#define BN254_H

#include "../../mont/src/mnt4753_fq.cuh"
#include "../../mont/src/mnt4753_fr.cuh"
#include "curve_xyzz.cuh"

namespace mnt4753
{
  using Element = mnt4753_fq::Element; // base field for curves
  using BaseNumber = mont::Number<Element::LIMBS>; // number for base field
  using Field = mnt4753_fr::Element; // field for scalars
  using Number = mont::Number<Field::LIMBS>; // number for scalar field

  namespace device_constants
  {
    constexpr __device__ Element b = Element(BaseNumber(BIG_INTEGER_CHUNKS24(0x19f48, 0x3a5ca38b, 0x8d306223, 0xebeb7591, 0x51b835e3, 0x477a5600, 0x81a3dd76, 0x44095968, 0x88696442, 0x003f5f9d, 0x6626f99c, 0x37f11ab9, 0x082ed0a8, 0x3dae31be, 0x4adf7a3e, 0x41c5734c, 0x18600804, 0xd0bcdc36, 0x122b02a6, 0x17b0220b, 0xf4144ce4, 0x02eb4f2a, 0x25171e93, 0x506fb062)));
    constexpr __device__ Element b3 = Element(BaseNumber(BIG_INTEGER_CHUNKS24(0x1544c, 0x53f06280, 0x874c0625, 0xe5fcc558, 0x85357309, 0x1f4fa226, 0xc77bc66f, 0xf2ec5b53, 0x8940ba7a, 0x2f7c43bc, 0xfed2a321, 0x651d60ef, 0xb65ca30a, 0xd4da89ff, 0x7080536f, 0x2ea95d73, 0x786a7e3b, 0x8551eb59, 0x6f70e70f, 0x11881a42, 0x15023e07, 0xaf07ca3f, 0xb22493fd, 0xa8921124)));
    constexpr __device__ Element a = Element(BaseNumber(BIG_INTEGER_CHUNKS24(0xf68f, 0x3d91c485, 0x2a8abf66, 0x3ff3432a, 0x1f48fdb6, 0x70cbd118, 0x80e99397, 0xfb194c42, 0xb3168605, 0xa5e014c4, 0x6418776e, 0x26670ab2, 0x3c1e9b15, 0x9e063ad1, 0xda4d3928, 0x42112ede, 0xf2b13033, 0x8f116c03, 0x2f87c941, 0x9a28ae5d, 0x239a638c, 0xb4068d0d, 0x3151d957, 0xb3b8de84)));
  }

  namespace host_constants
  {
    constexpr Element b = Element(BaseNumber(BIG_INTEGER_CHUNKS24(0x19f48, 0x3a5ca38b, 0x8d306223, 0xebeb7591, 0x51b835e3, 0x477a5600, 0x81a3dd76, 0x44095968, 0x88696442, 0x003f5f9d, 0x6626f99c, 0x37f11ab9, 0x082ed0a8, 0x3dae31be, 0x4adf7a3e, 0x41c5734c, 0x18600804, 0xd0bcdc36, 0x122b02a6, 0x17b0220b, 0xf4144ce4, 0x02eb4f2a, 0x25171e93, 0x506fb062)));
    constexpr Element b3 = Element(BaseNumber(BIG_INTEGER_CHUNKS24(0x1544c, 0x53f06280, 0x874c0625, 0xe5fcc558, 0x85357309, 0x1f4fa226, 0xc77bc66f, 0xf2ec5b53, 0x8940ba7a, 0x2f7c43bc, 0xfed2a321, 0x651d60ef, 0xb65ca30a, 0xd4da89ff, 0x7080536f, 0x2ea95d73, 0x786a7e3b, 0x8551eb59, 0x6f70e70f, 0x11881a42, 0x15023e07, 0xaf07ca3f, 0xb22493fd, 0xa8921124)));
    constexpr Element a = Element(BaseNumber(BIG_INTEGER_CHUNKS24(0xf68f, 0x3d91c485, 0x2a8abf66, 0x3ff3432a, 0x1f48fdb6, 0x70cbd118, 0x80e99397, 0xfb194c42, 0xb3168605, 0xa5e014c4, 0x6418776e, 0x26670ab2, 0x3c1e9b15, 0x9e063ad1, 0xda4d3928, 0x42112ede, 0xf2b13033, 0x8f116c03, 0x2f87c941, 0x9a28ae5d, 0x239a638c, 0xb4068d0d, 0x3151d957, 0xb3b8de84)));
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