#pragma once

#include "../../mont/src/field.cuh"
#include <iostream>

#ifdef __CUDA_ARCH__
#define likely(x) (__builtin_expect((x), 1))
#define unlikely(x) (__builtin_expect((x), 0))
#else
#define likely(x) (x) [[likely]]
#define unlikely(x) (x) [[unlikely]]
#endif 

namespace curve
{
    using mont::u32;
    using mont::usize;

    template <class Params, class Element>
    struct EC {
        struct PointXYZZ;
        struct PointAffine;

        // we assume that no points in pointaffine are identity
        struct PointAffine {
            static const usize N_WORDS = 2 * Element::LIMBS;

            Element x, y;

            friend std::ostream& operator<<(std::ostream &os, const PointAffine &p) {
                os << "{\n";
                os << "  .x = " << p.x << ",\n";
                os << "  .y = " << p.y << ",\n";
                os << "}";
                return os;
            }

            friend std::istream& operator>>(std::istream &is, PointAffine &p) {
                is >> p.x.n >> p.y.n;
                return is;
            }

            __host__ __device__ __forceinline__ PointAffine() {}
            __host__ __device__ __forceinline__ PointAffine(Element x, Element y) : x(x), y(y) {}

            __device__ __host__ __forceinline__ PointAffine neg() const & {
                return PointAffine(x, y.neg());
            }

            static __host__ __device__ __forceinline__ PointAffine load(const u32 *p) {
                auto x = Element::load(p);
                auto y = Element::load(p + Element::LIMBS);
                return PointAffine(x, y);
            }
            __host__ __device__ __forceinline__ void store(u32 *p) {
                x.store(p);
                y.store(p + Element::LIMBS);
            }

            __host__ __device__ __forceinline__
                PointAffine
                operator=(const PointAffine &rhs) &
            {
                if(this != &rhs) {
                    x = rhs.x;
                    y = rhs.y;
                }
                return *this;
            }

            static __device__ __host__ __forceinline__ PointAffine identity() {
                return PointAffine(Element::zero(), Element::zero());
            }

            __device__ __host__ __forceinline__ bool is_identity() const & {
                return y.is_zero();
            }

            __device__ __host__ __forceinline__ bool operator==(const PointAffine &rhs) const & {
                return x == rhs.x && y == rhs.y;
            }

            __device__ __host__ __forceinline__ bool is_on_curve() const & {
                Element t0, t1;
                t0 = x.square();
                if (!Params::a().is_zero()) t0 = t0 + Params::a();
                t0 = t0 * x;
                t0 = t0 + Params::b();
                t1 = y.square();
                t0 = t1 - t0;
                return t0.is_zero();
            }

            __device__ __forceinline__ bool is_on_curve_pre() const & {
                Element t0, t1;
                t0 = x.square_pre();
                if (!Params::a().is_zero()) t0 = t0.add_pre(Params::a());
                t0 = t0.mul_pre(x);
                t0 = t0.add_pre(Params::b());
                t1 = y.square_pre();
                t0 = t1.sub_pre(t0);
                return t0.is_zero();
            }

            __device__ __host__ __forceinline__ PointXYZZ to_point() const& {
                if unlikely(is_identity()) return PointXYZZ::identity();
                return PointXYZZ(x, y, Element::one(), Element::one());
            }

            __device__ __forceinline__ PointXYZZ to_point_pre() const& {
                return PointXYZZ(x, y, Element::one(), Element::one());
            }

            // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#doubling-mdbl-2008-s-1
            __device__ __host__ __forceinline__ PointXYZZ add_self() const& {
                auto u = y + y;
                auto v = u.square();
                auto w = u * v;
                auto s = x * v;
                auto x2 = x.square();
                auto m = x2 + x2 + x2;
                if (!Params::a().is_zero()) m = m + Params::a();
                auto x3 = m.square() - s - s;
                auto y3 = m * (s - x3) - w * y;
                return PointXYZZ(x3, y3, v, w);
            }

            __device__ __forceinline__ PointXYZZ add_self_pre() const& {
                auto u = y.add_pre(y);
                auto v = u.square_pre();
                auto w = u.mul_pre(v);
                auto s = x.mul_pre(v);
                auto x2 = x.square_pre();
                auto m = x2.add_pre(x2).add_pre(x2);
                if (!Params::a().is_zero()) m = m.add_pre(Params::a());
                auto x3 = m.square_pre().sub_pre(s).sub_pre(s);
                auto y3 = m.mul_pre(s.sub_pre(x3)).sub_pre(w.mul_pre(y));
                return PointXYZZ(x3, y3, v, w);
            }

            __host__ __device__ void device_print() const &
            {
                printf(
                    "{ x = %x %x %x %x %x %x %x %x\n, y = %x %x %x %x %x %x %x %x}\n",
                    x.n.limbs[7], x.n.limbs[6], x.n.limbs[5], x.n.limbs[4], x.n.limbs[3], x.n.limbs[2], x.n.limbs[1], x.n.limbs[0], 
                    y.n.limbs[7], y.n.limbs[6], y.n.limbs[5], y.n.limbs[4], y.n.limbs[3], y.n.limbs[2], y.n.limbs[1], y.n.limbs[0]
                );
            }
         };


        //  https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html
        //  x=X/ZZ
        //  y=Y/ZZZ
        //  ZZ^3=ZZZ^2
        struct PointXYZZ {
            static const usize N_WORDS = 4 * Element::LIMBS;
            Element x, y, zz, zzz;

            __host__ __device__ __forceinline__ PointXYZZ() {};
            __host__ __device__ __forceinline__ PointXYZZ(Element x, Element y, Element zz, Element zzz) : x(x), y(y), zz(zz), zzz(zzz) {}

            static __host__ __device__ __forceinline__ PointXYZZ load(const u32 *p) {
                auto x = Element::load(p);
                auto y = Element::load(p + Element::LIMBS);
                auto zz = Element::load(p + Element::LIMBS * 2);
                auto zzz = Element::load(p + Element::LIMBS * 3);
                return PointXYZZ(x, y, zz, zzz);
            }
            __host__ __device__ __forceinline__ void store(u32 *p) {
                x.store(p);
                y.store(p + Element::LIMBS);
                zz.store(p + Element::LIMBS * 2);
                zzz.store(p + Element::LIMBS * 3);
            }
            __host__ __device__ __forceinline__
                PointXYZZ
                operator=(const PointXYZZ &rhs) &
            {
                if(this != &rhs) {
                    x = rhs.x;
                    y = rhs.y;
                    zz = rhs.zz;
                    zzz = rhs.zzz;
                }
                return *this;
            }

            static constexpr __device__ __host__ __forceinline__ PointXYZZ identity() {
                return PointXYZZ(Element::zero(), Element::zero(), Element::zero(), Element::one());
            }

            __device__ __host__ __forceinline__ bool is_identity() const & {
                return zz.is_zero();
            }

            __device__ __forceinline__ bool is_identity_pre() const & {
                return zz.is_zero();
            }

            __device__ __host__ __forceinline__ PointXYZZ neg() const & {
                return PointXYZZ(x, y.neg(), zz, zzz);
            }

            __device__ __forceinline__ PointXYZZ neg_pre() const & {
                return PointXYZZ(x, y.neg(), zz, zzz);
            }

            __host__ __device__ __forceinline__ bool operator==(const PointXYZZ &rhs) const & {
                if (zz.is_zero() != rhs.zz.is_zero())
                    return false;
                // auto lhs = normalized();
                // auto rhs = rhs_.normalized();
                // printf("lhs:\n");
                // lhs.device_print();
                // printf("rhs:\n");
                // rhs.device_print();
                auto x1 = x * rhs.zz;
                auto x2 = rhs.x * zz;
                auto y1 = y * rhs.zzz;
                auto y2 = rhs.y * zzz;
                // printf(
                //     "thread %d\n{ x1 = %x %x %x %x %x %x %x %x\n, x2 = %x %x %x %x %x %x %x %x\n, y1 = %x %x %x %x %x %x %x %x\n, y2 = %x %x %x %x %x %x %x %x }\n",
                //     threadIdx.x, x1.n.limbs[7], x1.n.limbs[6], x1.n.limbs[5], x1.n.limbs[4], x1.n.limbs[3], x1.n.limbs[2], x1.n.limbs[1], x1.n.limbs[0], 
                //     x2.n.limbs[7], x2.n.limbs[6], x2.n.limbs[5], x2.n.limbs[4], x2.n.limbs[3], x2.n.limbs[2], x2.n.limbs[1], x2.n.limbs[0], 
                //     y1.n.limbs[7], y1.n.limbs[6], y1.n.limbs[5], y1.n.limbs[4], y1.n.limbs[3], y1.n.limbs[2], y1.n.limbs[1], y1.n.limbs[0], 
                //     y2.n.limbs[7], y2.n.limbs[6], y2.n.limbs[5], y2.n.limbs[4], y2.n.limbs[3], y2.n.limbs[2], y2.n.limbs[1], y2.n.limbs[0]
                // );
                return x1 == x2 && y1 == y2;
            }

            __device__ __forceinline__ bool eq_pre(const PointXYZZ &rhs) const & {
                if (zz.is_zero() != rhs.zz.is_zero())
                    return false;
                // auto lhs = normalized_pre();
                // auto rhs = rhs_.normalized_pre();
                auto x1 = x.mul_pre(rhs.zz);
                auto x2 = rhs.x.mul_pre(zz);
                auto y1 = y.mul_pre(rhs.zzz);
                auto y2 = rhs.y.mul_pre(zzz);
                return x1 == x2 && y1 == y2;
            }

            // x = X/ZZ
            // y = Y/ZZZ
            // ZZ^3 = ZZZ^2
            // y^2 = x^3 + a*x + b
            // Y^2/ZZZ^2 = X^3/ZZ^3 + a*X/ZZ + b
            // Y^2 = X^3 + a*X*ZZ^2 + b*ZZ^3
            __host__ __device__ __forceinline__ bool is_on_curve() const & {
                // auto self = normalized();
                auto y2 = y.square();
                auto x3 = x.square() * x;
                auto zz2 = zz.square();
                auto zz3 = zz * zz2;
                auto zzz2 = zzz.square();
                if (zz3 != zzz2) return false;
                Element a_x_zz2;
                if (Params::a().is_zero()) a_x_zz2 = Element::zero();
                else a_x_zz2 = Params::a() * x * zz2;
                auto b_zz3 = Params::b() * zz3;
                return y2 == x3 + a_x_zz2 + b_zz3;
            }

            __device__ __forceinline__ bool is_on_curve_pre() const & {
                // auto self = normalized_pre();
                auto y2 = y.square_pre();
                auto x3 = x.square_pre().mul_pre(x);
                auto zz2 = zz.square_pre();
                auto zz3 = zz.mul_pre(zz2);
                auto zzz2 = zzz.square_pre();
                if (zz3 != zzz2) return false;
                Element a_x_zz2;
                if (Params::a().is_zero()) a_x_zz2 = Element::zero();
                else a_x_zz2 = Params::a().mul_pre(x).mul_pre(zz2);
                auto b_zz3 = Params::b().mul_pre(zz3);
                return y2 == x3.add_pre(a_x_zz2).add_pre(b_zz3);
            }

            __device__ __host__ __forceinline__ bool is_elements_lt_2m() const &
            {
                return x.lt_2m() && y.lt_2m() && zz.lt_2m() && zzz.lt_2m();
            }

            // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#scaling-z
            __device__ __host__ __forceinline__ PointAffine to_affine() const & {
                // auto self = normalized();
                auto A = zzz.invert();
                auto B = (zz * A).square();
                auto X3 = x * B;
                auto Y3 = y * A;
                return PointAffine(X3, Y3);
            }

            __device__ __forceinline__ PointAffine to_affine_pre() const & {
                // auto self = normalized_pre();
                // self.device_print();
                auto A = zzz.invert_pre();
                // printf(
                //     "A = %x %x %x %x %x %x %x %x\n", A.n.limbs[7], A.n.limbs[6], A.n.limbs[5], A.n.limbs[4], A.n.limbs[3], A.n.limbs[2], A.n.limbs[1], A.n.limbs[0]
                // );
                auto C = zz.mul_pre(A);
                auto B = C.square_pre();
                // printf(
                //     "B = %x %x %x %x %x %x %x %x\n", B.n.limbs[7], B.n.limbs[6], B.n.limbs[5], B.n.limbs[4], B.n.limbs[3], B.n.limbs[2], B.n.limbs[1], B.n.limbs[0]
                // );
                auto X3 = x.mul_pre(B);
                auto Y3 = y.mul_pre(A);
                return PointAffine(X3, Y3);
            }

            __device__ __host__ __forceinline__ PointXYZZ add(const PointXYZZ &rhs) const &
            {
                if unlikely(this->is_identity()) return rhs;
                if unlikely(rhs.is_identity()) return *this;
                auto u1 = x * rhs.zz;
                auto u2 = rhs.x * zz;
                auto s1 = y * rhs.zzz;
                auto s2 = rhs.y * zzz;
                auto p = u2 - u1;
                auto r = s2 - s1;
                if unlikely(p.is_zero() && r.is_zero()) {
                    return this->self_add();
                }
                auto pp = p.square();            
                auto ppp = p * pp; 
                auto q = u1 * pp;
                auto x3 = r.square() - ppp - q - q;
                auto y3 = r * (q - x3) - s1 * ppp;
                auto zz3 = zz * rhs.zz * pp;
                auto zzz3 = zzz * rhs.zzz * ppp;
                return PointXYZZ(x3, y3, zz3, zzz3);
            }

            __host__ __device__ void device_print() const &
            {
                printf(
                    "{ x = %x %x %x %x %x %x %x %x\n, y = %x %x %x %x %x %x %x %x\n, zz = %x %x %x %x %x %x %x %x\n, zzz = %x %x %x %x %x %x %x %x }\n",
                    x.n.limbs[7], x.n.limbs[6], x.n.limbs[5], x.n.limbs[4], x.n.limbs[3], x.n.limbs[2], x.n.limbs[1], x.n.limbs[0], 
                    y.n.limbs[7], y.n.limbs[6], y.n.limbs[5], y.n.limbs[4], y.n.limbs[3], y.n.limbs[2], y.n.limbs[1], y.n.limbs[0], 
                    zz.n.limbs[7], zz.n.limbs[6], zz.n.limbs[5], zz.n.limbs[4], zz.n.limbs[3], zz.n.limbs[2], zz.n.limbs[1], zz.n.limbs[0], 
                    zzz.n.limbs[7], zzz.n.limbs[6], zzz.n.limbs[5], zzz.n.limbs[4], zzz.n.limbs[3], zzz.n.limbs[2], zzz.n.limbs[1], zzz.n.limbs[0]
                );
            }

            // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-add-2008-s
            __device__ __host__ __forceinline__ PointXYZZ operator + (const PointXYZZ &rhs) const & {
                return add(rhs);
            }

            __device__ __forceinline__ PointXYZZ add_pre(const PointXYZZ &rhs) const & {
                if unlikely(this->is_identity()) return rhs;
                if unlikely(rhs.is_identity()) return *this;
                auto u1 = x.mul_pre(rhs.zz);
                auto u2 = rhs.x.mul_pre(zz);
                auto s1 = y.mul_pre(rhs.zzz);
                auto s2 = rhs.y.mul_pre(zzz);
                auto p = u2.sub_pre(u1);
                auto r = s2.sub_pre(s1);
                if unlikely(p.is_zero() && r.is_zero()) {
                    return this->self_add_pre();
                }
                auto pp = p.square_pre();
                auto ppp = p.mul_pre(pp);
                auto q = u1.mul_pre(pp);
                auto x3 = r.square_pre().sub_pre(ppp).sub_pre(q).sub_pre(q);
                auto y3 = r.mul_pre(q.sub_pre(x3)).sub_pre(s1.mul_pre(ppp));
                auto zz3 = zz.mul_pre(rhs.zz).mul_pre(pp);
                auto zzz3 = zzz.mul_pre(rhs.zzz).mul_pre(ppp);
                return PointXYZZ(x3, y3, zz3, zzz3);
            }

            __device__ __host__ __forceinline__ PointXYZZ operator - (const PointXYZZ &rhs) const & {
                return *this + rhs.neg();
            }

            __device__ __forceinline__ PointXYZZ sub_pre(const PointXYZZ &rhs) const & {
                return add_pre(rhs.neg_pre());
            }

            // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-madd-2008-s
            __device__ __host__ __forceinline__ PointXYZZ add(const PointAffine &rhs) const & {
                if unlikely(this->is_identity()) return rhs.to_point();
                if unlikely(rhs.is_identity()) return *this;
                auto u2 = rhs.x * zz;
                auto s2 = rhs.y * zzz;
                auto p = u2 - x;
                auto r = s2 - y;
                if unlikely(p.is_zero() && r.is_zero()) {
                    return rhs.add_self();
                }
                auto pp = p.square();
                auto ppp = p * pp;
                auto q = x * pp;
                auto x3 = r.square() - ppp - q - q;
                auto y3 = r * (q - x3) - y * ppp;
                auto zz3 = zz * pp;
                auto zzz3 = zzz * ppp;
                return PointXYZZ(x3, y3, zz3, zzz3);
            }

            __device__ __host__ __forceinline__ PointXYZZ operator + (const PointAffine &rhs) const & {
                return add(rhs);
            }

            __device__ __forceinline__ PointXYZZ add_pre(const PointAffine &rhs) const & {
                if unlikely(this->is_identity()) return rhs.to_point();
                if unlikely(rhs.is_identity()) return *this;
                auto u2 = rhs.x.mul_pre(zz);
                auto s2 = rhs.y.mul_pre(zzz);
                auto p = u2.sub_pre(x);
                auto r = s2.sub_pre(y);
                if unlikely(p.is_zero() && r.is_zero()) {
                    return rhs.add_self_pre();
                }
                auto pp = p.square_pre();
                auto ppp = p.mul_pre(pp);
                auto q = x.mul_pre(pp);
                auto x3 = r.square_pre().sub_pre(ppp).sub_pre(q).sub_pre(q);
                auto y3 = r.mul_pre(q.sub_pre(x3)).sub_pre(y.mul_pre(ppp));
                auto zz3 = zz.mul_pre(pp);
                auto zzz3 = zzz.mul_pre(ppp);
                return PointXYZZ(x3, y3, zz3, zzz3);
           }

            __device__ __host__ __forceinline__ PointXYZZ operator - (const PointAffine &rhs) const & {
                return *this + rhs.neg();
            }

            __device__ __forceinline__ PointXYZZ sub_pre(const PointAffine &rhs) const & {
                return add_pre(rhs.neg());
            }

            // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#doubling-dbl-2008-s-1
            __device__ __host__ __forceinline__ PointXYZZ self_add() const & {
                if unlikely(zz.is_zero()) return *this;
                auto u = y + y;
                auto v = u.square();
                auto w = u * v;
                auto s = x * v;
                auto x2 = x.square();
                auto m = x2 + x2 + x2;
                if (!Params::a().is_zero()) m = m + (Params::a() * zz.square());
                auto x3 = m.square() - s - s;
                auto y3 = m * (s - x3) - w * y;
                auto zz3 = v * zz;
                auto zzz3 = w * zzz;
                return PointXYZZ(x3, y3, zz3, zzz3);
            }

            __device__ __host__ __forceinline__ PointXYZZ self_add_pre() const & {
                if unlikely(zz.is_zero()) return *this;
                auto u = y.add_pre(y);
                auto v = u.square_pre();
                auto w = u.mul_pre(v);
                auto s = x.mul_pre(v);
                auto x2 = x.square_pre();
                auto m = x2.add_pre(x2).add_pre(x2);
                if (!Params::a().is_zero()) m = m.add_pre(Params::a() * zz.square_pre());
                auto x3 = m.square_pre().sub_pre(s).sub_pre(s);
                auto y3 = m.mul_pre(s.sub_pre(x3)).sub_pre(w.mul_pre(y));
                auto zz3 = v.mul_pre(zz);
                auto zzz3 = w.mul_pre(zzz);
                return PointXYZZ(x3, y3, zz3, zzz3);
            }

            static __device__ __host__ __forceinline__ void multiple_iter(const PointXYZZ &p, bool &found_one, PointXYZZ &res, u32 n) {
                for (int i = 31; i >= 0; i--) {
                    if (found_one) res = res.self_add();
                    if ((n >> i) & 1) {
                        found_one = true;
                        res = res + p;
                    }
                }
            }

            __device__ __host__ __forceinline__ PointXYZZ multiple(u32 n) const & {
                auto res = identity();
                bool found_one = false;
                multiple_iter(*this, found_one, res, n);
                return res;
            }

            __device__ __forceinline__ PointXYZZ multiple_pre(u32 n) const & {
                auto res = identity();
                bool found_one = false;
                for (int i = 31; i >= 0; i--) {
                    if (found_one) res = res.self_add_pre();
                    if ((n >> i) & 1) {
                        found_one = true;
                        res = res.add_pre(*this);
                    }
                }
                return res;
            }

            __device__ __host__ __forceinline__ PointXYZZ shuffle_down(const u32 delta) const & {
                PointXYZZ res;
                #pragma unroll
                for (usize i = 0; i < Element::LIMBS; i++) {
                    res.x.n.limbs[i] = __shfl_down_sync(0xFFFFFFFF, x.n.limbs[i], delta);
                    res.y.n.limbs[i] = __shfl_down_sync(0xFFFFFFFF, y.n.limbs[i], delta);
                    res.zz.n.limbs[i] = __shfl_down_sync(0xFFFFFFFF, zz.n.limbs[i], delta);
                    res.zzz.n.limbs[i] = __shfl_down_sync(0xFFFFFFFF, zzz.n.limbs[i], delta);
                }
                return res;
            }
        };
    };
}