#ifndef STORAGE_H
#define STORAGE_H

namespace mont
{

  using u8 = u_int8_t;
  using u16 = u_int16_t;
  using u32 = u_int32_t;
  using i32 = int32_t;
  using u64 = u_int64_t;
  using i64 = int64_t;
  using i64 = int64_t;
  using usize = size_t;

  // A reference to a storage in memory, strided
  template <usize STRIDE>
  struct StridedReference
  {
    u32 *p;

    __device__ __host__ __forceinline__ u32 &operator[](usize i)
    {
      return *(p + i * STRIDE);
    }
    __device__ __host__ __forceinline__ const u32 &operator[](usize i) const
    {
      return *(p + i * STRIDE);
    }

    __device__ __host__ __forceinline__ StridedReference<STRIDE> to_ref()
    {
      return *this;
    }

    __device__ __host__ __forceinline__ const StridedReference<STRIDE> to_ref() const
    {
      return *this;
    }

    __device__ __host__ __forceinline__ StridedReference<STRIDE> operator+(int k)
    {
      return StridedReference<STRIDE>{.p = p + k * STRIDE};
    }

    __device__ __host__ __forceinline__ const StridedReference<STRIDE> operator+(int k) const
    {
      return StridedReference<STRIDE>{.p = p + k * STRIDE};
    }

    template <usize N>
    __device__ __host__ __forceinline__ void set_zero()
    {
      if (STRIDE == 1)
        memset(p, 0, N * sizeof(u32));
      else
#pragma unroll
        for (usize i = 0; i < N; i++)
          p[i * STRIDE] = 0;
    }

    template <usize N>
    __device__ __host__ __forceinline__ void store(u32 *out, u32 stride = 1) const &
    {
#ifdef __CUDA_ARCH__
      if (stride == 1 && STRIDE == 1 && N % 4 == 0)
      {
#pragma unroll
        for (usize i = 0; i < N / 4; i++)
        {
          reinterpret_cast<uint4 *>(out)[i] = reinterpret_cast<const uint4 *>(p)[i];
        }
      }
      else if (stride == 1 && STRIDE == 1 && N % 2 == 0)
      {
#pragma unroll
        for (usize i = 0; i < N / 2; i++)
        {
          reinterpret_cast<uint2 *>(out)[i] = reinterpret_cast<const uint2 *>(p)[i];
        }
      }
      else
#endif
      {
#pragma unroll
        for (usize i = 0; i < N; i++)
          out[i * stride] = p[i * STRIDE];
      }
    }
  };

  // An array for storage
  template <usize N>
  struct __align__(16) ContinuousStorage
  {
    u32 limbs[N];

    __device__ __host__ ContinuousStorage() {}

    constexpr ContinuousStorage(const std::initializer_list<uint32_t> &values) : limbs{}
    {
      size_t i = 0;
      for (auto value : values)
      {
        if (i >= N)
          break;
        limbs[i++] = value;
      }
    }

    __device__ __host__ __forceinline__ u32 &operator[](usize i)
    {
      return limbs[i];
    }

    __device__ __host__ __forceinline__ const u32 &operator[](usize i) const
    {
      return limbs[i];
    }

    __device__ __host__ __forceinline__ StridedReference<1> operator+(int k)
    {
      return StridedReference<1>{.p = (u32 *)limbs + k};
    }

    __device__ __host__ __forceinline__ const StridedReference<1> operator+(int k) const
    {
      return StridedReference<1>{.p = (u32 *)limbs + k};
    }

    template <usize N1>
    __device__ __host__ __forceinline__ void set_zero()
    {
      memset(limbs, 0, sizeof(u32) * N1);
    }

    __device__ __host__ __forceinline__ StridedReference<1> to_ref()
    {
      return StridedReference<1>{.p = limbs};
    }

    __device__ __host__ __forceinline__ const StridedReference<1> to_ref() const
    {
      return StridedReference<1>{.p = (u32 *)limbs};
    }

    static __device__ __host__ __forceinline__
        ContinuousStorage
        load(const u32 *p, u32 stride = 1)
    {
      ContinuousStorage r;
#ifdef __CUDA_ARCH__
      if (stride == 1 && N % 4 == 0)
      {
#pragma unroll
        for (usize i = 0; i < N / 4; i++)
        {
          reinterpret_cast<uint4 *>(r.limbs)[i] = reinterpret_cast<const uint4 *>(p)[i];
        }
      }
      else if (stride == 1 && N % 2 == 0)
      {
#pragma unroll
        for (usize i = 0; i < N / 2; i++)
        {
          reinterpret_cast<uint2 *>(r.limbs)[i] = reinterpret_cast<const uint2 *>(p)[i];
        }
      }
      else
#endif
      {
#pragma unroll
        for (usize i = 0; i < N; i++)
          r.limbs[i] = p[i * stride];
      }
      return r;
    }

    template <usize N1>
    __device__ __host__ __forceinline__ void store(u32 * p, u32 stride = 1) const &
    {
      const auto ref = StridedReference<1>{.p = (u32 *)limbs};
      ref.template store<N1>(p, stride);
    }
  };

  template <usize N, typename Src, typename Dst>
  __host__ __device__ __forceinline__ void storage_copy(Dst dst, Src src)
  {
    for (usize i = 0; i < N; i++)
      dst[i] = src[i];
  }

  template <usize N>
  __host__ __device__ __forceinline__ void storage_copy(StridedReference<1> dst, const StridedReference<1> src)
  {
    memcpy(dst.p, src.p, N * sizeof(u32));
  }
}

#endif