#ifndef BN254_H
#define BN254_H

#include "../../mont/src/bls12381_fq.cuh"
#include "curve_xyzz.cuh"

namespace bls12381
{
  using mont::u32;
  using mont::usize;
  using Params1 = bls12381_fq::Params;

  const u32 b_h[12] = BIG_INTEGER_CHUNKS12(0x9d64551, 0x3d83de7e, 0x8ec9733b, 0xbf78ab2f, 0xb1d37ebe, 0xe6ba24d7, 0x478fe97a, 0x6b0a807f, 0x53cc0032, 0xfc34000a, 0xaa270000, 0x000cfff3);
  const u32 b3_h[12] = BIG_INTEGER_CHUNKS12(0x381be09, 0x7f0bb4e1, 0x6140b1fc, 0xfb1e54b7, 0xb10330b7, 0xc0a95bc6, 0x6f7ee9ce, 0x4a6e8b59, 0xdcb8009a, 0x43480020, 0x44760000, 0x0027552e);
  const u32 a_h[12] = BIG_INTEGER_CHUNKS12(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

  const __device__ u32 b_d[12] = BIG_INTEGER_CHUNKS12(0x9d64551, 0x3d83de7e, 0x8ec9733b, 0xbf78ab2f, 0xb1d37ebe, 0xe6ba24d7, 0x478fe97a, 0x6b0a807f, 0x53cc0032, 0xfc34000a, 0xaa270000, 0x000cfff3);
  const __device__ u32 b3_d[12] = BIG_INTEGER_CHUNKS12(0x381be09, 0x7f0bb4e1, 0x6140b1fc, 0xfb1e54b7, 0xb10330b7, 0xc0a95bc6, 0x6f7ee9ce, 0x4a6e8b59, 0xdcb8009a, 0x43480020, 0x44760000, 0x0027552e);
  const __device__ u32 a_d[12] = BIG_INTEGER_CHUNKS12(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

  struct Params
  {

    template <usize LIMBS>
    static __device__ __host__ __forceinline__
        mont::Element<Params1, LIMBS>
        a(int i, int j)
    {
        mont::Element<Params1, LIMBS> r;
        for(int k=0; k<j-i; ++k) {
#ifdef __CUDA_ARCH__
            r.n._limbs[k] = a_d[i+k];
#else
            r.n._limbs[k] = a_h[i+k];
#endif
        }
        return r;
    }

    static __device__ __host__ __forceinline__ bool a_is_zero() {
        return true;
    }

    template <usize LIMBS>
    static __device__ __host__ __forceinline__
        mont::Element<Params1, LIMBS>
        b(int i, int j)
    {
        mont::Element<Params1, LIMBS> r;
        for(int k=0; k<j-i; ++k) {
#ifdef __CUDA_ARCH__
            r.n._limbs[k] = b_d[i+k];
#else
            r.n._limbs[k] = b_h[i+k];
#endif
        }
        return r;
    }

    template <usize LIMBS>
    static __device__ __host__ __forceinline__
        mont::Element<Params1, LIMBS>
        b3(int i, int j)
    {
        mont::Element<Params1, LIMBS> r;
        for(int k=0; k<j-i; ++k) {
#ifdef __CUDA_ARCH__
            r.n._limbs[k] = b3_d[i+k];
#else
            r.n._limbs[k] = b3_h[i+k];
#endif
        }
        return r;
    }

    static constexpr __device__ __host__ __forceinline__
        bool
        allow_lazy_modulo()
    {
        return true;
    }
  };

  using Point = curve::EC<Params>::PointXYZZ<Params1::LIMBS/TPI>;
  using PointAffine = curve::EC<Params>::PointAffine<Params1::LIMBS/TPI>;
  using PointAll = curve::EC<Params>::PointXYZZ<Params1::LIMBS>;
  using PointAffineAll = curve::EC<Params>::PointAffine<Params1::LIMBS>;
}

#endif