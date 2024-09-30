#ifndef CURVE_H
#define CURVE_H

#include "../../mont/src/field.cuh"

namespace curve
{
  using mont::u32;

  template <class Params, class Element>
  struct PointAffine;

  // A point (x z^(-1), y z^(-1)) on curve
  //   y^2 = x^3 + a x + b
  // z = 0 markes an identity point.
  //
  // Algorithms from
  //   Complete Addition Formulas for Prime Order Elliptic Curves
  template <class Params, class Element>
  struct Point
  {
    Element x, y, z;

    __host__ __device__ __forceinline__ Point() {}
    __host__ __device__ __forceinline__ Point(Element x, Element y, Element z) : x(x), y(y), z(z) {}
    static __host__ __device__ __forceinline__ Point load(const u32 *p)
    {
      auto x = Element::load(p);
      auto y = Element::load(p + Element::LIMBS);
      auto z = Element::load(p + Element::LIMBS * 2);
      return Point(x, y, z);
    }
    __host__ __device__ __forceinline__ void store(u32 *p)
    {
      x.store(p);
      y.store(p + Element::LIMBS);
      z.store(p + Element::LIMBS * 2);
    }

    __device__ __host__ __forceinline__ PointAffine<Params, Element> to_affine() const &;

    static __device__ __host__ __forceinline__
        Point
        identity()
    {
      return Point(Element::zero(), Element::one(), Element::zero());
    }

    __device__ __host__ __forceinline__ bool is_identity() const &
    {
      return z.is_zero();
    }

    __device__ __host__ __forceinline__ bool operator==(const Point &rhs) const &
    {
      if (is_identity() && rhs.is_identity())
        return true;
      auto x1 = x * rhs.z;
      auto y1 = y * rhs.z;
      auto x2 = rhs.x * z;
      auto y2 = rhs.y * z;
      return x1 == x2 && y1 == y2;
    }

    __device__ __host__ __forceinline__
        Point
        neg() const &
    {
      return Point(x, y.neg(), z);
    }

    __device__ __host__ __forceinline__ bool is_on_curve() const &
    {
      Element t0, t1;
      t0 = Params::a() * x;
      t1 = Params::b() * z;
      t0 = t0 + t1;
      t0 = z * t0;
      t1 = y.square();
      t0 = t0 - t1;
      t0 = t0 * z;
      t1 = x.square();
      t1 = t1 * x;
      t0 = t0 + t1;
      return t0.is_zero();
    }

    __device__ __host__ __forceinline__ Point self_add() const &
    {
      Element t0, t1, t2, t3, x3, y3, z3;
      t0 = x.square();
      t1 = y.square();
      t2 = z.square();
      t3 = x * y;
      t3 = t3 + t3;
      z3 = x * z;
      z3 = z3 + z3;
      x3 = Params::a() * z3;
      y3 = Params::b3() * t2;
      y3 = x3 + y3;
      x3 = t1 - y3;
      y3 = t1 + y3;
      y3 = x3 * y3;
      x3 = t3 * x3;
      z3 = Params::b3() * z3;
      t2 = Params::a() * t2;
      t3 = t0 - t2;
      t3 = Params::a() * t3;
      t3 = t3 + z3;
      z3 = t0 + t0;
      t0 = z3 + t0;
      t0 = t0 + t2;
      t0 = t0 * t3;
      y3 = y3 + t0;
      t2 = y * z;
      t2 = t2 + t2;
      t0 = t2 * t3;
      x3 = x3 - t0;
      z3 = t2 * t1;
      z3 = z3 + z3;
      z3 = z3 + z3;
      return Point(x3, y3, z3);
    }

    __device__ __host__ __forceinline__ Point operator+(const Point &rhs) const &
    {
      if (is_identity())
        return rhs;
      if (rhs.is_identity())
        return *this;
      Element t0, t1, t2, t3, t4, t5, x3, y3, z3;
      t0 = x * rhs.x;
      t1 = y * rhs.y;
      t2 = z * rhs.z;
      t3 = x + y;
      t4 = rhs.x + rhs.y;
      t3 = t3 * t4;
      t4 = t0 + t1;
      t3 = t3 - t4;
      t4 = x + z;
      t5 = rhs.x + rhs.z;
      t4 = t4 * t5;
      t5 = t0 + t2;
      t4 = t4 - t5;
      t5 = y + z;
      x3 = rhs.y + rhs.z;
      t5 = t5 * x3;
      x3 = t1 + t2;
      t5 = t5 - x3;
      z3 = Params::a() * t4;
      x3 = Params::b3() * t2;
      z3 = x3 + z3;
      x3 = t1 - z3;
      z3 = t1 + z3;
      y3 = x3 * z3;
      t1 = t0 + t0;
      t1 = t1 + t0;
      t2 = Params::a() * t2;
      t4 = Params::b3() * t4;
      t1 = t1 + t2;
      t2 = t0 - t2;
      t2 = Params::a() * t2;
      t4 = t4 + t2;
      t0 = t1 * t4;
      y3 = y3 + t0;
      t0 = t5 * t4;
      x3 = t3 * x3;
      x3 = x3 - t0;
      t0 = t3 * t1;
      z3 = t5 * z3;
      z3 = z3 + t0;
      return Point(x3, y3, z3);
    }

    __device__ __host__ __forceinline__ Point operator+(const PointAffine<Params, Element> &rhs) const &;

    static __device__ __host__ __forceinline__ void multiple_iter(const Point &p, bool &found_one, Point &res, u32 n)
    {
      for (int i = 31; i >= 0; i--)
      {
        if (found_one)
          res = res.self_add();
        if ((n >> i) & 1)
        {
          found_one = true;
          res = res + p;
        }
      }
    }

    template <mont::usize N>
    __device__ __host__ __forceinline__ Point multiple(const mont::Number<N> &n) const &
    {
      auto res = identity();
      bool found_one = false;
#pragma unroll
      for (int i = N - 1; i >= 0; i--)
        multiple_iter(*this, found_one, res, n.limbs[i]);
      return res;
    }

    __device__ __host__ __forceinline__ Point multiple(u32 n) const &
    {
      auto res = identity();
      bool found_one = false;
      multiple_iter(*this, found_one, res, n);
      return res;
    }
  };

  template <class Params, class Element>
  std::ostream &
  operator<<(std::ostream &os, const Point<Params, Element> &p)
  {
    os << "{\n";
    os << "  .x = " << p.x << ",\n";
    os << "  .y = " << p.y << ",\n";
    os << "  .z = " << p.z << ",\n";
    os << "}";
    return os;
  }

  const u32 WORDS_PER_POINT_AFFINE = 16;

  // A point (x, y) on curve. Identity marked by (0, 0).
  template <class Params, class Element>
  struct PointAffine
  {
    Element x, y;

    static const u32 N_WORDS = 16;

    __host__ __device__ __forceinline__ PointAffine() {}
    __host__ __device__ __forceinline__ PointAffine(Element x, Element y) : x(x), y(y) {}
    static __host__ __device__ __forceinline__ PointAffine load(const u32 *p)
    {
      auto x = Element::load(p);
      auto y = Element::load(p + Element::LIMBS);
      return PointAffine(x, y);
    }
    __host__ __device__ __forceinline__ void store(u32 *p)
    {
      x.store(p);
      y.store(p + Element::LIMBS);
    }

    __device__ __host__ __forceinline__ Point<Params, Element> to_projective() const &;

    static __device__ __host__ __forceinline__
        PointAffine
        identity()
    {
      return PointAffine(Element::zero(), Element::zero());
    }

    __device__ __host__ __forceinline__ bool is_identity() const &
    {
      return x.is_zero() && y.is_zero();
    }

    __device__ __host__ __forceinline__ bool operator==(const PointAffine &rhs) const &
    {
      return x == rhs.x && y == rhs.y;
    }

    __device__ __host__ __forceinline__
        PointAffine
        neg() const &
    {
      return PointAffine(x, y.neg());
    }

    __device__ __host__ __forceinline__ bool is_on_curve() const &
    {
      Element t0, t1;
      t0 = x.square();
      t0 = t0 + Params::a();
      t0 = t0 * x;
      t0 = t0 + Params::b();
      t1 = y.square();
      t0 = t1 - t0;
      return t0.is_zero();
    }
  };

  template <class Params, class Element>
  __device__ __host__ __forceinline__ PointAffine<Params, Element> Point<Params, Element>::to_affine() const &
  {
    auto zinv = z.invert();
    auto x1 = x * zinv;
    auto y1 = y * zinv;
    return PointAffine<Params, Element>(x1, y1);
  }

  template <class Params, class Element>
  __device__ __host__ __forceinline__ Point<Params, Element> PointAffine<Params, Element>::to_projective() const &
  {
    return Point<Params, Element>(x, y, Element::one());
  }

  template <class Params, class Element>
  __device__ __host__ __forceinline__
      Point<Params, Element>
      Point<Params, Element>::operator+(const PointAffine<Params, Element> &rhs) const &
  {
    if (rhs.is_identity())
      return *this;
    if (is_identity())
      return rhs.to_projective();
    Element t0, t1, t2, t3, t4, t5, x3, y3, z3;
    t0 = x * rhs.x;
    t1 = y * rhs.y;
    t3 = rhs.x + rhs.y;
    t4 = x + y;
    t3 = t3 * t4;
    t4 = t0 + t1;
    t3 = t3 - t4;
    t4 = rhs.x * z;
    t4 = t4 + x;
    t5 = rhs.y * z;
    t5 = t5 + y;
    z3 = Params::a() * t4;
    x3 = Params::b3() * z;
    z3 = x3 + z3;
    x3 = t1 - z3;
    z3 = t1 + z3;
    y3 = x3 * z3;
    t1 = t0 + t0;
    t1 = t1 + t0;
    t2 = Params::a() * z;
    t4 = Params::b3() * t4;
    t1 = t1 + t2;
    t2 = t0 - t2;
    t2 = Params::a() * t2;
    t4 = t4 + t2;
    t0 = t1 * t4;
    y3 = y3 + t0;
    t0 = t5 * t4;
    x3 = t3 * x3;
    x3 = x3 - t0;
    t0 = t3 * t1;
    z3 = t5 * z3;
    z3 = z3 + t0;
    return Point<Params, Element>(x3, y3, z3);
  }

  template <class Params, class Element>
  std::ostream &
  operator<<(std::ostream &os, const PointAffine<Params, Element> &p)
  {
    os << "{\n";
    os << "  .x = " << p.x << ",\n";
    os << "  .y = " << p.y << ",\n";
    os << "}";
    return os;
  }

  template <class Params, class Element>
  std::istream &
  operator>>(std::istream &is, PointAffine<Params, Element> &p)
  {
    is >> p.x >> p.y;
    return is;
  }
}
#endif