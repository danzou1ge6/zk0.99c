#ifndef MONT2_H
#define MONT2_H

#include "./field.cuh"

namespace mont {
  template <class Once, class Params>
  struct Element2 {
    using OnceType = Once;
    using ParamsType = Params;
    static const usize LIMBS = Once::LIMBS;

    Once c0, c1;

    __host__ __device__ __forceinline__ Element2() {}
    constexpr __host__ __device__ __forceinline__ Element2(Number<LIMBS> c0, Number<LIMBS> c1) : c0(c0), c1(c1) {}
    constexpr __host__ __device__ __forceinline__ Element2(Once c0, Once c1) : c0(c0), c1(c1) {}

    static __host__ __device__ __forceinline__
        Element2
        load(const u32 *p, u32 stride = 1)
    {
      Element2 r;
      r.c0 = Once::load(p, stride);
      r.c1 = Once::load(p + sizeof(Once) * stride, stride);
      return r;
    }

    __host__ __device__ __forceinline__ void store(u32 *p, u32 stride = 1) const &
    {
      c0.store(p, stride);
      c1.store(p + sizeof(Once) * stride, stride);
    }

    static __host__ __device__ __forceinline__
        Element2
        zero()
    {
      return Element2(Once::zero(), Once::zero());
    }

    __host__ __device__ __forceinline__ constexpr bool is_zero() const &
    {
      return c0.is_zero() && c1.is_zero();
    }

    __host__ __device__ __forceinline__ bool operator==(const Element2 &rhs) const &
    {
      return c0 == rhs.c0 && c1 == rhs.c1;
    }

    __host__ __device__ __forceinline__ bool operator!=(const Element2 &rhs) const &
    {
      return!(*this == rhs);
    }

    static __device__ __host__ __forceinline__
        Element2
        one()
    {
      return Element2(Once::one(), Once::zero());
    }

    // Lazy Modulo not implemented for Fq2
    template <bool MODULO = true>
    __host__ __device__ __forceinline__
        Element2
        mul(const Element2 &rhs) const &
    {
      return *this * rhs;
    }

    __host__ __device__ __forceinline__
        Element2
        operator*(const Element2 &rhs) const &
    {
      Element2 r;
      Element t0 = c0 * rhs.c0;
      Element t1 = c1 * rhs.c1;

      r.c0 = t0 + Params::non_residue() * t1;
      r.c1 = (c0 + rhs.c0) * (c1 + rhs.c1) - t0 - t1;
      return r;
    }

    template <bool MODULO = true>
    __host__ __device__ __forceinline__
        Element2
        square() const &
    {
      return mul<MODULO>(*this);
    }

    __host__ __device__ __forceinline__
        Element2
        operator+(const Element2 &rhs) const &
    {
      Element2 r;
      r.c0 = c0 + rhs.c0;
      r.c1 = c1 + rhs.c1;
      return r;
    }

    __host__ __device__ __forceinline__
        Element2
        operator-(const Element2 &rhs) const &
    {
      Element2 r;
      r.c0 = c0 - rhs.c0;
      r.c1 = c1 - rhs.c1;
      return r;
    }

    // Lazy Modulo not implemented for Fq2, so modulo-2m subtract is implemented as module-m to ensure consistency
    __device__ __host__ __forceinline__
        Element2
        sub_modulo_mm2(const Element2 &rhs) const &
    {
      return *this - rhs;
    }

    __host__ __device__ __forceinline__
        Element2
        add_modulo_mm2(const Element2 &rhs) const &
    {
      return *this + rhs;
    }

    // Lazy Modulo not implemented for Fq2, so invariant 0 <= c0, c1 < m always holds,
    // and there is no need for modulo-m
    __device__ __host__ __forceinline__
        Element2
        modulo_m() const &
    {
      return *this;
    }

    __host__ __device__ __forceinline__
        Element2
        neg() const &
    {
      if (is_zero())
        return *this;
      return Element2(c0.neg(), c1.neg());
    }

    static __host__ __device__ __forceinline__
        Element2
        from_number(const Number<LIMBS> c0, const Number<LIMBS> c1)
    {
      return Element2(Once::from_number(c0), Once::from_number(c1));
    }

    __host__ __device__ __forceinline__
        void
        to_number(Number<LIMBS> &c0, Number<LIMBS> &c1) const &
    {
      c0 = this->c0.to_number();
      c1 = this->c1.to_number();
    }

    // Helper function for `pow`
    static __host__ __device__ __forceinline__ void
    pow_iter(const Element2 &a, bool &found_one, Element2 &res, u32 p, u32 deg = 31)
    {
#pragma unroll
      for (int i = deg; i >= 0; i--)
      {
        if (found_one)
          res = res.square();
        if ((p >> i) & 1)
        {
          found_one = true;
          res = res * a;
        }
      }
    }

    // Field power
    __host__ __device__ __forceinline__
        Element2
        pow(const Number<LIMBS> &p) const &
    {
      auto res = one();
      bool found_one = false;
#pragma unroll
      for (int i = LIMBS - 1; i >= 0; i--)
        pow_iter(*this, found_one, res, p.limbs[i]);
      return res;
    }

    __host__ __device__ __forceinline__ Element2 pow(u32 p, u32 deg = 31) const &
    {
      auto res = one();
      bool found_one = false;
      pow_iter(*this, found_one, res, p, deg);
      return res;
    }

    // Field inversion
    __host__ __device__ __forceinline__ Element2 invert() const &
    {
      Once t0 = c0.square();
      Once t1 = c1.square();
      Once t2 = t0 - Params::non_residue() * t1;
      Once t3 = t2.invert();
      Once a0 = c0 * t3;
      Once a1 = (c1 * t3).neg();

      return Element2(a0, a1);
    }

    static __host__ __forceinline__
        Element2
        host_random()
    {
      return Element2(Once::host_random(), Once::host_random());
    }

    __device__ __forceinline__ Element2 shuffle_down(const u32 delta, u32 mask = 0xFFFFFFFF) const &
    {
      Element2 res;
      res.c0 = c0.shuffle_down(delta, mask);
      res.c1 = c1.shuffle_down(delta, mask);
      return res;
    }
  };

  template <class Once, class Params>
  __forceinline__ std::ostream &
  operator<<(std::ostream &os, const Element2<Once, Params> &e)
  {
    os << e.c0 << "+" << e.c1 << "x";
    return os;
  }

  template <class Once, class Params>
  __forceinline__ std::istream &
  operator<<(std::istream &is, Element2<Once, Params> &e)
  {
    is >> e.c0 >> e.c1;
    return is;
  }
}

#endif