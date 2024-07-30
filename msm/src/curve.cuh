#ifndef CURVE_H
#define CURVE_H

#include "../../mont/src/mont.cuh"

namespace curve256
{
  using mont256::Element;
  using mont256::Number;
  using mont256::u32;

  // A point on curve (x z^(-1), y z^(-1)).
  // z = 0 markes an identity point.
  struct Point
  {
    Element x, y, z;

    __host__ __device__ __forceinline__ Point() {}
    __host__ __device__ __forceinline__ Point(Element x, Element y, Element z) : x(x), y(y), z(z) {}

  };

  std::ostream &
  operator<<(std::ostream &os, const Point &p)
  {
    os << "{\n";
    os << "  .x = " << p.x << ",\n";
    os << "  .y = " << p.y << ",\n";
    os << "  .z = " << p.z << ",\n";
    os << "}";
    return os;
  }


  // A point on curve. Identity marked by (0, 0).
  struct PointAffine
  {
    Element x, y;

    __host__ __device__ __forceinline__ PointAffine() {}
    __host__ __device__ __forceinline__ PointAffine(Element x, Element y) : x(x), y(y) {}
  };

  std::ostream &
  operator<<(std::ostream &os, const PointAffine &p)
  {
    os << "{\n";
    os << "  .x = " << p.x << ",\n";
    os << "  .y = " << p.y << ",\n";
    os << "}";
    return os;
  }

  // Y^2 = X^3 + aX + b
  struct Curve
  {
    mont256::Env field;
    // b3 = 3 b
    Element a, b, b3;

    __device__ __forceinline__ Curve(mont256::Env field, Element a, Element b, Element b3) : field(field), a(a), b(b), b3(b3)
    {
    }

    __device__ __forceinline__ PointAffine to_affine(const Point &p)
    {
      auto zinv = field.invert(p.z);
      auto x = field.mul(p.x, zinv);
      auto y = field.mul(p.y, zinv);
      return PointAffine(x, y);
    }

    __device__ __forceinline__ Point from_affine(const PointAffine &p)
    {
      return Point(p.x, p.y, field.one());
    }

    __device__ __host__ __forceinline__ PointAffine identity_affine()
    {
      return PointAffine(field.zero(), field.zero());
    }

    __device__ __host__ __forceinline__ Point identity()
    {
      return Point(field.zero(), field.one(), field.zero());
    }

    __device__ __host__ __forceinline__ bool is_identity(const Point &p)
    {
      return field.is_zero(p.z);
    }

    __device__ __host__ __forceinline__ bool is_identity(const PointAffine &p)
    {
      return field.is_zero(p.x) && field.is_zero(p.y);
    }

    __device__ __forceinline__ bool eq(const Point &a, const Point &b)
    {
      auto x1 = field.mul(a.x, b.z);
      auto y1 = field.mul(a.y, b.z);
      auto x2 = field.mul(b.x, a.z);
      auto y2 = field.mul(b.y, a.z);
      return is_identity(a) && is_identity(b) || !is_identity(a) && !is_identity(b) && x1 == x2 && y1 == y2;
    }

    __device__ __forceinline__ bool eq(const PointAffine &a, const PointAffine &b)
    {
      return a.x == b.x && a.y == b.y;
    }

    __device__ __forceinline__ Point neg(const Point &p)
    {
      return Point(p.x, field.neg(p.y), p.z);
    }

    __device__ __forceinline__ PointAffine neg(const PointAffine &p)
    {
      return PointAffine(p.x, field.neg(p.y));
    }

    __device__ __forceinline__ bool is_on_curve(const Point &p)
    {
      Element t0, t1;
      t0 = field.mul(a, p.x);
      t1 = field.mul(b, p.z);
      t0 = field.add(t0, t1);
      t0 = field.mul(p.z, t0);
      t1 = field.square(p.y);
      t0 = field.sub(t0, t1);
      t0 = field.mul(t0, p.z);
      t1 = field.mul(p.x, p.x);
      t1 = field.mul(t1, p.x);
      t0 = field.add(t0, t1);
      return field.is_zero(t0);
    }

    __device__ __forceinline__ bool is_on_curve(const PointAffine &p)
    {
      Element t0, t1;
      t0 = field.square(p.x);
      t0 = field.add(t0, a);
      t0 = field.mul(t0, p.x);
      t0 = field.add(t0, b);
      t1 = field.square(p.y);
      t0 = field.sub(t1, t0);
      return field.is_zero(t0);
    }

    // A alternative name for double, which is not a valid function name
    // Ref. Complete Addition Formulas for Prime Order Elliptic Curves, Algorithm 3.
    __device__ __forceinline__ Point self_add(const Point &p)
    {
      Element t0, t1, t2, t3, x3, y3, z3;
      t0 = field.square(p.x);
      t1 = field.square(p.y);
      t2 = field.square(p.z);
      t3 = field.mul(p.x, p.y);
      t3 = field.add(t3, t3);
      z3 = field.mul(p.x, p.z);
      z3 = field.add(z3, z3);
      x3 = field.mul(a, z3);
      y3 = field.mul(b3, t2);
      y3 = field.add(x3, y3);
      x3 = field.sub(t1, y3);
      y3 = field.add(t1, y3);
      y3 = field.mul(x3, y3);
      x3 = field.mul(t3, x3);
      z3 = field.mul(b3, z3);
      t2 = field.mul(a, t2);
      t3 = field.sub(t0, t2);
      t3 = field.mul(a, t3);
      t3 = field.add(t3, z3);
      z3 = field.add(t0, t0);
      t0 = field.add(z3, t0);
      t0 = field.add(t0, t2);
      t0 = field.mul(t0, t3);
      y3 = field.add(y3, t0);
      t2 = field.mul(p.y, p.z);
      t2 = field.add(t2, t2);
      t0 = field.mul(t2, t3);
      x3 = field.sub(x3, t0);
      z3 = field.mul(t2, t1);
      z3 = field.add(z3, z3);
      z3 = field.add(z3, z3);
      return Point(x3, y3, z3);
    }

    // Adding two points.
    // Ref. Complete Addition Formulas for Prime Order Elliptic Curves, Algorithm 1.
    __device__ __forceinline__ Point add(const Point &p1, const Point &p2)
    {
      Element t0, t1, t2, t3, t4, t5, x3, y3, z3;
      t0 = field.mul(p1.x, p2.x);
      t1 = field.mul(p1.y, p2.y);
      t2 = field.mul(p1.z, p2.z);
      t3 = field.add(p1.x, p1.y);
      t4 = field.add(p2.x, p2.y);
      t3 = field.mul(t3, t4);
      t4 = field.add(t0, t1);
      t3 = field.sub(t3, t4);
      t4 = field.add(p1.x, p1.z);
      t5 = field.add(p2.x, p2.z);
      t4 = field.mul(t4, t5);
      t5 = field.add(t0, t2);
      t4 = field.sub(t4, t5);
      t5 = field.add(p1.y, p1.z);
      x3 = field.add(p2.y, p2.z);
      t5 = field.mul(t5, x3);
      x3 = field.add(t1, t2);
      t5 = field.sub(t5, x3);
      z3 = field.mul(a, t4);
      x3 = field.mul(b3, t2);
      z3 = field.add(x3, z3);
      x3 = field.sub(t1, z3);
      z3 = field.add(t1, z3);
      y3 = field.mul(x3, z3);
      t1 = field.add(t0, t0);
      t1 = field.add(t1, t0);
      t2 = field.mul(a, t2);
      t4 = field.mul(b3, t4);
      t1 = field.add(t1, t2);
      t2 = field.sub(t0, t2);
      t2 = field.mul(a, t2);
      t4 = field.add(t4, t2);
      t0 = field.mul(t1, t4);
      y3 = field.add(y3, t0);
      t0 = field.mul(t5, t4);
      x3 = field.mul(t3, x3);
      x3 = field.sub(x3, t0);
      t0 = field.mul(t3, t1);
      z3 = field.mul(t5, z3);
      z3 = field.add(z3, t0);
      return Point(x3, y3, z3);
    }

    // Addint one point and one affine point.
    // Ref. Complete Addition Formulas for Prime Order Elliptic Curves, Algorithm 2.
    __device__ __forceinline__ Point add(const Point &p1, const PointAffine &p2)
    {
      Element t0, t1, t2, t3, t4, t5, x3, y3, z3;
      t0 = field.mul(p1.x, p2.x);
      t1 = field.mul(p1.y, p2.y);
      t3 = field.add(p2.x, p2.y);
      t4 = field.add(p1.x, p1.y);
      t3 = field.mul(t3, t4);
      t4 = field.add(t0, t1);
      t3 = field.sub(t3, t4);
      t4 = field.mul(p2.x, p1.z);
      t4 = field.add(t4, p1.x);
      t5 = field.mul(p2.y, p1.z);
      t5 = field.add(t5, p1.y);
      z3 = field.mul(a, t4);
      x3 = field.mul(b3, p1.z);
      z3 = field.add(x3, z3);
      x3 = field.sub(t1, z3);
      z3 = field.add(t1, z3);
      y3 = field.mul(x3, z3);
      t1 = field.add(t0, t0);
      t1 = field.add(t1, t0);
      t2 = field.mul(a, p1.z);
      t4 = field.mul(b3, t4);
      t1 = field.add(t1, t2);
      t2 = field.sub(t0, t2);
      t2 = field.mul(a, t2);
      t4 = field.add(t4, t2);
      t0 = field.mul(t1, t4);
      y3 = field.add(y3, t0);
      t0 = field.mul(t5, t4);
      x3 = field.mul(t3, x3);
      x3 = field.sub(x3, t0);
      t0 = field.mul(t3, t1);
      z3 = field.mul(t5, z3);
      z3 = field.add(z3, t0);
      return Point(x3, y3, z3);
    }

    __device__ __forceinline__ void multiple_iter(const Point &p, bool &found_one, Point &res, u32 n)
    {
      for (int i = 31; i >= 0; i--)
      {
        if (found_one)
          res = self_add(res);
        if ((n >> i) & 1)
        {
          found_one = true;
          res = add(res, p);
        }
      }
    }

    __device__ __forceinline__ Point multiple(const Point &p, const Number& n)
    {
      auto res = identity();
      bool found_one = false;
      multiple_iter(p, found_one, res, n.c7);
      multiple_iter(p, found_one, res, n.c6);
      multiple_iter(p, found_one, res, n.c5);
      multiple_iter(p, found_one, res, n.c4);
      multiple_iter(p, found_one, res, n.c3);
      multiple_iter(p, found_one, res, n.c2);
      multiple_iter(p, found_one, res, n.c1);
      multiple_iter(p, found_one, res, n.c0);
      return res;
    }

    __device__ __forceinline__ Point multiple(const Point &p, u32 n)
    {
      auto res = identity();
      bool found_one = false;
      multiple_iter(p, found_one, res, n);
      return res;
    }
  };
}

#endif