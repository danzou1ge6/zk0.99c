#ifndef BN254_H
#define BN254_H

#include "../../mont/src/bn254_scalar.cuh"
#include "curve_xyzz.cuh"

namespace bn254
{
  using mont::u32;
  using mont::usize;
  using Params1 = bn254_scalar::Params;

  const u32 b_h[8] = BIG_INTEGER_CHUNKS8(0x2a1f6744, 0xce179d8e, 0x334bea4e, 0x696bd284, 0x1f6ac17a, 0xe15521b9, 0x7a17caa9, 0x50ad28d7);
  const u32 b3_h[8] = BIG_INTEGER_CHUNKS8(0x1d9598e8, 0xa7e39857, 0x2943337e, 0x3940c6d1, 0x2f3d6f4d, 0xd31bd011, 0xf60647ce, 0x410d7ff7);
  const u32 a_h[8] = BIG_INTEGER_CHUNKS8(0, 0, 0, 0, 0, 0, 0, 0);

  const __device__ u32 b_d[8] = BIG_INTEGER_CHUNKS8(0x2a1f6744, 0xce179d8e, 0x334bea4e, 0x696bd284, 0x1f6ac17a, 0xe15521b9, 0x7a17caa9, 0x50ad28d7);
  const __device__ u32 b3_d[8] = BIG_INTEGER_CHUNKS8(0x1d9598e8, 0xa7e39857, 0x2943337e, 0x3940c6d1, 0x2f3d6f4d, 0xd31bd011, 0xf60647ce, 0x410d7ff7);
  const __device__ u32 a_d[8] = BIG_INTEGER_CHUNKS8(0, 0, 0, 0, 0, 0, 0, 0);

//   namespace device_constants
//   {
//     constexpr __device__ Element b = Element(Number(BIG_INTEGER_CHUNKS8(
//         0x2a1f6744, 0xce179d8e, 0x334bea4e, 0x696bd284, 0x1f6ac17a, 0xe15521b9, 0x7a17caa9, 0x50ad28d7)));
//     constexpr __device__ Element b3 = Element(Number(BIG_INTEGER_CHUNKS8(
//         0x1d9598e8, 0xa7e39857, 0x2943337e, 0x3940c6d1, 0x2f3d6f4d, 0xd31bd011, 0xf60647ce, 0x410d7ff7)));
//     constexpr __device__ Element a = Element(Number(BIG_INTEGER_CHUNKS8(
//         0, 0, 0, 0, 0, 0, 0, 0)));
//   }

//   namespace host_constants
//   {
//     constexpr Element b = Element(Number(BIG_INTEGER_CHUNKS8(
//         0x2a1f6744, 0xce179d8e, 0x334bea4e, 0x696bd284, 0x1f6ac17a, 0xe15521b9, 0x7a17caa9, 0x50ad28d7)));
//     constexpr Element b3 = Element(Number(BIG_INTEGER_CHUNKS8(
//         0x1d9598e8, 0xa7e39857, 0x2943337e, 0x3940c6d1, 0x2f3d6f4d, 0xd31bd011, 0xf60647ce, 0x410d7ff7)));
//     constexpr Element a = Element(Number(BIG_INTEGER_CHUNKS8(
//         0, 0, 0, 0, 0, 0, 0, 0)));
//   }

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

//     static __device__ __host__ __forceinline__
//         mont::Element<Params1, Params1::LIMBS>
//         a_all()
//     {
//         mont::Element<Params1, Params1::LIMBS> r;
//         for(int k=0; k<Params1::LIMBS; ++k) {
// #ifdef __CUDA_ARCH__
//             r.n._limbs[k] = a_d[k];
// #else
//             r.n._limbs[k] = a_h[k];
// #endif
//         }
//         return r;
//     }

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

//     static __device__ __host__ __forceinline__
//         mont::Element<Params1, Params1::LIMBS>
//         b_all()
//     {
//         mont::Element<Params1, Params1::LIMBS> r;
//         for(int k=0; k<Params1::LIMBS; ++k) {
// #ifdef __CUDA_ARCH__
//             r.n._limbs[k] = b_d[k];
// #else
//             r.n._limbs[k] = b_h[k];
// #endif
//         }
//         return r;
//     }

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

//     static __device__ __host__ __forceinline__
//         mont::Element<Params1, Params1::LIMBS>
//         b3_all()
//     {
//         mont::Element<Params1, Params1::LIMBS> r;
//         for(int k=0; k<Params1::LIMBS; ++k) {
// #ifdef __CUDA_ARCH__
//             r.n._limbs[k] = b3_d[k];
// #else
//             r.n._limbs[k] = b3_h[k];
// #endif
//         }
//         return r;
//     }

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
//   using PointEager = curve::EC<Params<false>, Element>::PointXYZZ;
//   using PointAffineEager = curve::EC<Params<false>, Element>::PointAffine;

//   __host__ __device__ PointEager eager_from_lazy(const Point &lazy)
//   {
//     auto lazy_n = lazy.normalized();
//     return PointEager(lazy_n.x, lazy_n.y, lazy_n.zz, lazy_n.zzz);
//   }

//   __host__ __device__ PointAffineEager eager_from_lazy(const PointAffine &lazy)
//   {
//     return PointAffineEager(lazy.x, lazy.y);
//   }

//   __host__ __device__ Point lazy_from_eager(const PointEager &eager)
//   {
//     return Point(eager.x, eager.y, eager.zz, eager.zzz);
//   }

//   __host__ __device__ PointAffine lazy_from_eager(const PointAffineEager &eager)
//   {
//     return PointAffine(eager.x, eager.y);
//   }
}

#endif