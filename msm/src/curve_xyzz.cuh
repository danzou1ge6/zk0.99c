// modified from ML's code
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

            static __device__ __host__ __forceinline__ PointAffine identity() {
                return PointAffine(Element::zero(), Element::zero());
            }

            __device__ __host__ __forceinline__ bool is_identity() const & {
                return x.is_zero() && y.is_zero();
            }

            __device__ __host__ __forceinline__ bool operator==(const PointAffine &rhs) const & {
                return x == rhs.x && y == rhs.y;
            }

            __device__ __host__ __forceinline__ bool is_on_curve() const & {
                Element t0, t1;
                t0 = x.square();
                if constexpr (!Params::a().is_zero()) t0 = t0 + Params::a();
                t0 = t0 * x;
                t0 = t0 + Params::b();
                t1 = y.square();
                t0 = t1 - t0;
                return t0.is_zero();
            }

            __device__ __host__ __forceinline__ PointXYZZ to_point() const& {
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
                if constexpr (!Params::a().is_zero()) m = m + Params::a();
                auto x3 = m.square() - s - s;
                auto y3 = m * (s - x3) - w * y;
                return PointXYZZ(x3, y3, v, w);
            }
         };


        //  https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html
        //  x=X/ZZ
        //  y=Y/ZZZ
        //  ZZ^3=ZZZ^2
        struct PointXYZZ {
            static const usize N_WORDS = 4 * Element::LIMBS;
            Element x, y, zz, zzz;

            __host__ __device__ __forceinline__ PointXYZZ() = default;
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

            static constexpr __device__ __host__ __forceinline__ PointXYZZ identity() {
                return PointXYZZ(Element::zero(), Element::zero(), Element::zero(), Element::one());
            }

            __device__ __host__ __forceinline__ bool is_identity() const & {
                return zz.is_zero();
            }

            __device__ __host__ __forceinline__ PointXYZZ neg() const & {
                return PointXYZZ(x, y.neg(), zz, zzz);
            }

            __device__ __host__ __forceinline__ bool operator==(const PointXYZZ &rhs) const & {
                if (zz.is_zero() != rhs.zz.is_zero())
                    return false;
                auto x1 = x * rhs.zz;
                auto x2 = rhs.x * zz;
                auto y1 = y * rhs.zzz;
                auto y2 = rhs.y * zzz;
                return x1 == x2 && y1 == y2;
            }

            // x = X/ZZ
            // y = Y/ZZZ
            // ZZ^3 = ZZZ^2
            // y^2 = x^3 + a*x + b
            // Y^2/ZZZ^2 = X^3/ZZ^3 + a*X/ZZ + b
            // Y^2 = X^3 + a*X*ZZ^2 + b*ZZ^3
            __device__ __host__ __forceinline__ bool is_on_curve() const & {
                auto y2 = y.square();
                auto x3 = x.square() * x;
                auto zz2 = zz.square();
                auto zz3 = zz * zz2;
                auto zzz2 = zzz.square();
                if (zz3 != zzz2) return false;
                Element a_x_zz2;
                if constexpr (Params::a().is_zero()) a_x_zz2 = Element::zero();
                else a_x_zz2 = Params::a() * x * zz2;
                auto b_zz3 = Params::b() * zz3;
                return y2 == x3 + a_x_zz2 + b_zz3;
            }

            // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#scaling-z
            __device__ __host__ __forceinline__ PointAffine to_affine() const & {
                auto A = zzz.invert();
                auto B = (zz * A).square();
                auto X3 = x * B;
                auto Y3 = y * A;
                return PointAffine(X3, Y3);
            }

            // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-add-2008-s
            __device__ __host__ __forceinline__ PointXYZZ operator + (const PointXYZZ &rhs) const & {
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

            __device__ __host__ __forceinline__ PointXYZZ operator - (const PointXYZZ &rhs) const & {
                return *this + rhs.neg();
            }

            // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-madd-2008-s
            __device__ __host__ __forceinline__ PointXYZZ operator + (const PointAffine &rhs) const & {
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

            __device__ __host__ __forceinline__ PointXYZZ operator - (const PointAffine &rhs) const & {
                return *this + rhs.neg();
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
                if constexpr (!Params::a().is_zero()) m = m + (Params::a() * zz.square());
                auto x3 = m.square() - s - s;
                auto y3 = m * (s - x3) - w * y;
                auto zz3 = v * zz;
                auto zzz3 = w * zzz;
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



        // // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#doubling-mdbl-2008-s-1
        // static  __device__ __forceinline__ PointXYZZ mdbl_2008_s_1(const PointAffine &point, const FD &fd) {
        //     const storage Y1 = point.y;                                        //  < 2
        //     const storage X1 = point.x;                                        //  < 2
        //     const storage U = fd.template dbl<2>(Y1);                          // U = 2*Y1      < 2
        //     const storage V = fd.template sqr<0>(U);                           // V = U^2       < 2
        //     const storage W = fd.template mul<0>(U, V);                        // W = U*V       < 2
        //     const storage S = fd.template mul<0>(X1, V);                       // S = X1*V      < 2
        //     const storage t0 = fd.template sqr<1>(X1);                         // t0 = X1^2     < 1
        //     const storage t1 = fd.template add<0>(t0, fd.template dbl<1>(t0)); // t1 = 3*t0     < 2
        //     const storage M = t1;                                              // M = t1+a      < 2
        //     const storage t2 = fd.template sqr<0>(M);                          // t2 = M^2      < 2
        //     const storage t3 = fd.template dbl<2>(S);                          // t3 = 2*S      < 2
        //     const storage X3 = fd.template sub<2>(t2, t3);                     // X3 = t2-t3    < 2
        //     const storage t4 = fd.template sub<2>(S, X3);                      // t4 = S-X3     < 2
        // #ifdef __CUDA_ARCH__
        //     // Y3 = M*t4 - W*Y1
        //     const storage_wide t5_wide = fd.template mul_wide<0>(W, Y1);   // < 4*mod^2 (Y1 may be in [0, 2*mod))
        //     const storage_wide t6_wide = fd.template mul_wide<0>(M, t4);   // < 4*mod^2
        //     storage_wide diff = fd.template sub_wide<4>(t6_wide, t5_wide); // < 4*mod^2
        //     fd.redc_wide_inplace(diff);                                    // < 2*mod, hi limbs 0
        //     const storage Y3 = diff.get_lo();                              // < 2*mod
        // #else
        //     const storage t5 = fd.template mul<0>(W, Y1);     // t5 = W*Y1     < 2
        //     const storage t6 = fd.template mul<0>(M, t4);     // t6 = M*t4     < 2
        //     const storage Y3 = fd.template sub<2>(t6, t5);    // Y3 = t6-t5    < 2
        // #endif
        //     const storage ZZ3 = V;  // ZZ3 = V       < 2
        //     const storage ZZZ3 = W; // ZZZ3 = W      < 2
        //     return {X3, Y3, ZZ3, ZZZ3};
        // }

        // static  __device__ __forceinline__ PointXYZZ dbl(const PointAffine &point, const FD &fd) { return mdbl_2008_s_1(point, fd); }

        // // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#doubling-dbl-2008-s-1
        // template <bool CHECK_ZERO = true> static  __device__ __forceinline__ PointXYZZ dbl_2008_s_1(const PointXYZZ &point, const FD &fd) {
        //     const storage X1 = point.x;     // < 2
        //     const storage Y1 = point.y;     // < 2
        //     const storage ZZ1 = point.zz;   // < 2
        //     const storage ZZZ1 = point.zzz; // < 2
        //     if (CHECK_ZERO) {
        //     if (unlikely(fd.is_zero(fd.reduce(ZZ1))))
        //         return point;
        //     }
        //     const storage U = fd.template dbl<2>(Y1);                          // U = 2*Y1       < 2
        //     const storage V = fd.template sqr<0>(U);                           // V = U^2        < 2
        //     const storage W = fd.template mul<0>(U, V);                        // W = U*V        < 2
        //     const storage S = fd.template mul<0>(X1, V);                       // S = X1*V       < 2
        //     const storage t0 = fd.template sqr<1>(X1);                         // t0 = X1^2      < 1
        //                                                                     // t1 = ZZ1^2          unused
        //                                                                     // t2 = a*t1           a=0 => t2 = 0
        //     const storage t3 = fd.template add<0>(t0, fd.template dbl<1>(t0)); // t3 = 3*t0      < 2
        //     const storage M = t3;                                              // M = t3+t2      < 2  t2 = 0 => M = t3
        //     const storage t4 = fd.template sqr<0>(M);                          // t4 = M^2       < 2
        //     const storage t5 = fd.template dbl<2>(S);                          // t5 = 2*S       < 2
        //     const storage X3 = fd.template sub<2>(t4, t5);                     // X3 = t4-t5     < 2
        //     const storage t6 = fd.template sub<2>(S, X3);                      // t6 = S-X3      < 2
        // #ifdef __CUDA_ARCH__
        //     // Y3 = M*t6 - W*Y1
        //     const storage_wide t7_wide = fd.template mul_wide<0>(W, Y1);   // < 4*mod^2 (Y1 may be in [0, 2*mod))
        //     const storage_wide t8_wide = fd.template mul_wide<0>(M, t6);   // < 4*mod^2
        //     storage_wide diff = fd.template sub_wide<4>(t8_wide, t7_wide); // < 4*mod^2
        //     fd.redc_wide_inplace(diff);                                    // < 2*mod, hi limbs 0
        //     const storage Y3 = diff.get_lo();                              // < 2*mod
        // #else
        //     const storage t7 = fd.template mul<0>(W, Y1);     // t7 = W*Y1      < 2
        //     const storage t8 = fd.template mul<0>(M, t6);     // t8 = M*t6      < 2
        //     const storage Y3 = fd.template sub<2>(t8, t7);    // Y3 = t8-t7     < 2
        // #endif
        //     const storage ZZ3 = fd.template mul<0>(V, ZZ1);   // ZZ3 = V*ZZ1    < 2
        //     const storage ZZZ3 = fd.template mul<0>(W, ZZZ1); // ZZZ3 = W*ZZZ1  < 2
        //     return {X3, Y3, ZZ3, ZZZ3};
        // }

        // template <bool CHECK_ZERO = true> static  __device__ __forceinline__ PointXYZZ dbl(const PointXYZZ &point, const FD &fd) {
        //     return dbl_2008_s_1<CHECK_ZERO>(point, fd);
        // }

        // // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-add-2008-s
        // template <bool CHECK_ZERO = true, bool CHECK_DOUBLE = true>
        // static  __device__ __forceinline__ PointXYZZ add_2008_s(const PointXYZZ &p1, const PointXYZZ &p2, const FD &fd) {
        //     const storage X1 = p1.x;     // < 2
        //     const storage Y1 = p1.y;     // < 2
        //     const storage ZZ1 = p1.zz;   // < 2
        //     const storage ZZZ1 = p1.zzz; // < 2
        //     const storage X2 = p2.x;     // < 2
        //     const storage Y2 = p2.y;     // < 2
        //     const storage ZZ2 = p2.zz;   // < 2
        //     const storage ZZZ2 = p2.zzz; // < 2
        //     if (CHECK_ZERO) {
        //     if (unlikely(fd.is_zero(fd.reduce(ZZ1))))
        //         return p2;
        //     if (unlikely(fd.is_zero(fd.reduce(ZZ2))))
        //         return p1;
        //     }
        //     const storage U1 = fd.template mul<0>(X1, ZZ2);  // U1 = X1*ZZ2   < 2
        //     const storage U2 = fd.template mul<0>(X2, ZZ1);  // U2 = X2*ZZ1   < 2
        //     const storage S1 = fd.template mul<0>(Y1, ZZZ2); // S1 = Y1*ZZZ2  < 2
        //     const storage S2 = fd.template mul<0>(Y2, ZZZ1); // S2 = Y2*ZZZ1  < 2
        //     const storage P = fd.template sub<2>(U2, U1);    // P = U2-U1     < 2
        //     const storage R = fd.template sub<2>(S2, S1);    // R = S2-S1     < 2
        //     if (CHECK_DOUBLE) {
        //     if (unlikely(fd.is_zero(fd.reduce(P))) && unlikely(fd.is_zero(fd.reduce(R))))
        //         return dbl<false>(p1, fd);
        //     }
        //     const storage PP = fd.template sqr<0>(P);       // PP = P^2        < 2
        //     const storage PPP = fd.template mul<0>(P, PP);  // PPP = P*PP      < 2
        //     const storage Q = fd.template mul<0>(U1, PP);   // Q = U1*PP       < 2
        //     const storage t0 = fd.template sqr<0>(R);       // t0 = R^2        < 2
        //     const storage t1 = fd.template dbl<2>(Q);       // t1 = 2*Q        < 2
        //     const storage t2 = fd.template sub<2>(t0, PPP); // t2 = t0-PPP     < 2
        //     const storage X3 = fd.template sub<2>(t2, t1);  // X3 = t2-t1      < 2
        //     const storage t3 = fd.template sub<2>(Q, X3);   // t3 = Q-X3       < 2
        // #ifdef __CUDA_ARCH__
        //     // Y3 = R*t3 - S1*PPP (requires R, t3, S1, PPP < 2*mod)
        //     const storage_wide t4_wide = fd.template mul_wide<0>(S1, PPP); // < 4*mod^2
        //     const storage_wide t5_wide = fd.template mul_wide<0>(R, t3);   // < 4*mod^2
        //     storage_wide diff = fd.template sub_wide<4>(t5_wide, t4_wide); // < 4*mod^2
        //     fd.redc_wide_inplace(diff);                                    // < 2*mod, hi limbs 0
        //     const storage Y3 = diff.get_lo();                              // < 2*mod
        // #else
        //     const storage t4 = fd.template mul<0>(S1, PPP);   // t4 = S1*PPP     < 2
        //     const storage t5 = fd.template mul<0>(R, t3);     // t5 = R*t3       < 2
        //     const storage Y3 = fd.template sub<2>(t5, t4);    // Y3 = t5-t4      < 2
        // #endif

        //     const storage t6 = fd.template mul<0>(ZZ2, PP);    // t6 = ZZ2*PP     < 2
        //     const storage ZZ3 = fd.template mul<0>(ZZ1, t6);   // ZZ3 = ZZ1*t6    < 2
        //     const storage t7 = fd.template mul<0>(ZZZ2, PPP);  // t7 = ZZZ2*PPP   < 2
        //     const storage ZZZ3 = fd.template mul<0>(ZZZ1, t7); // ZZZ3 = ZZZ1*t7  < 2
        //     return {X3, Y3, ZZ3, ZZZ3};
        // }

        // template <bool CHECK_ZERO = true, bool CHECK_DOUBLE = true>
        // static  __device__ __forceinline__ PointXYZZ add(const PointXYZZ &p1, const PointXYZZ &p2, const FD &fd) {
        //     return add_2008_s<CHECK_ZERO, CHECK_DOUBLE>(p1, p2, fd);
        // }

        // // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-madd-2008-s
        // template <bool CHECK_ZERO = true, bool CHECK_DOUBLE = true>
        // static  __device__ __forceinline__ PointXYZZ add_madd_2008_s(const PointXYZZ &p1, const PointAffine &p2, const FD &fd) {
        //     const storage X1 = p1.x;     // < 2
        //     const storage Y1 = p1.y;     // < 2
        //     const storage ZZ1 = p1.zz;   // < 2
        //     const storage ZZZ1 = p1.zzz; // < 2
        //     const storage X2 = p2.x;     // < 1
        //     const storage Y2 = p2.y;     // < 1
        //     if (CHECK_ZERO) {
        //     if (unlikely(fd.is_zero(fd.reduce(ZZ1))))
        //         return PointAffine::to_xyzz(p2, fd);
        //     }
        //     const storage U2 = fd.template mul<0>(X2, ZZ1);  // U2 = X2*ZZ1       < 2
        //     const storage S2 = fd.template mul<0>(Y2, ZZZ1); // S2 = Y2*ZZZ1      < 2
        //     const storage P = fd.template sub<2>(U2, X1);    // P = U2-X1         < 2
        //     const storage R = fd.template sub<2>(S2, Y1);    // R = S2-Y1         < 2
        //     if (CHECK_DOUBLE) {
        //     if (unlikely(fd.is_zero(fd.reduce(P))) && unlikely(fd.is_zero(fd.reduce(R))))
        //         return dbl(p2, fd);
        //     }
        //     const storage PP = fd.template sqr<0>(P);       // PP = P^2           < 2
        //     const storage PPP = fd.template mul<0>(P, PP);  // PPP = P*PP         < 2
        //     const storage Q = fd.template mul<0>(X1, PP);   // Q = X1*PP          < 2
        //     const storage t0 = fd.template sqr<0>(R);       // t0 = R^2           < 2
        //     const storage t1 = fd.template dbl<2>(Q);       // t1 = 2*Q           < 2
        //     const storage t2 = fd.template sub<2>(t0, PPP); // t2 = t0-PPP        < 2
        //     const storage X3 = fd.template sub<2>(t2, t1);  // X3 = t2-t1         < 2
        //     const storage t3 = fd.template sub<2>(Q, X3);   // t3 = Q-X3          < 2
        // #ifdef __CUDA_ARCH__
        //     // Y3 = R*t3-Y1*PPP
        //     const storage_wide t4_wide = fd.template mul_wide<0>(Y1, PPP); //         < 4*mod^2
        //     const storage_wide t5_wide = fd.template mul_wide<0>(R, t3);   //         < 4*mod^2
        //     storage_wide diff = fd.template sub_wide<4>(t5_wide, t4_wide); //         < 4*mod^2
        //     fd.redc_wide_inplace(diff);                                    //         < 2*mod, hi limbs 0
        //     const storage Y3 = diff.get_lo();                              //         < 2*mod
        // #else
        //     const storage t4 = fd.template mul<0>(Y1, PPP);   // t4 = Y1*PPP        < 2
        //     const storage t5 = fd.template mul<0>(R, t3);     // t5 = R*t3          < 2
        //     const storage Y3 = fd.template sub<2>(t5, t4);    // Y3 = t5-t4         < 2
        // #endif
        //     const storage ZZ3 = fd.template mul<0>(ZZ1, PP);    // ZZ3 = ZZ1*PP       < 2
        //     const storage ZZZ3 = fd.template mul<0>(ZZZ1, PPP); // ZZZ3 = ZZZ1*PPP    < 2
        //     return {X3, Y3, ZZ3, ZZZ3};
        // }

        // template <bool CHECK_ZERO = true, bool CHECK_DOUBLE = true>
        // static  __device__ __forceinline__ PointXYZZ add(const PointXYZZ &p1, const PointAffine &p2, const FD &fd) {
        //     return add_madd_2008_s<CHECK_ZERO, CHECK_DOUBLE>(p1, p2, fd);
        // }

        // static  __device__ __forceinline__ PointProjective add(const PointProjective &p1, const PointProjective &p2, const FD &fd) {
        //     const storage X1 = p1.x;                                                 //                   < 2
        //     const storage Y1 = p1.y;                                                 //                   < 2
        //     const storage Z1 = p1.z;                                                 //                   < 2
        //     const storage X2 = p2.x;                                                 //                   < 2
        //     const storage Y2 = p2.y;                                                 //                   < 2
        //     const storage Z2 = p2.z;                                                 //                   < 2
        //     const storage t00 = fd.template mul<0>(X1, X2);                          // t00 ← X1 · X2     < 2
        //     const storage t01 = fd.template mul<0>(Y1, Y2);                          // t01 ← Y1 · Y2     < 2
        //     const storage t02 = fd.template mul<0>(Z1, Z2);                          // t02 ← Z1 · Z2     < 2
        //     const storage t03 = fd.template add<0>(X1, Y1);                          // t03 ← X1 + Y1     < 4
        //     const storage t04 = fd.template add<0>(X2, Y2);                          // t04 ← X2 + Y2     < 4
        //     const storage t05 = fd.template mul<2>(t03, t04);                        // t03 ← t03 · t04   < 3
        //     const storage t06 = fd.template add<0>(t00, t01);                        // t06 ← t00 + t01   < 4
        //     const storage t07 = fd.template reduce<2>(fd.template sub<4>(t05, t06)); // t05 ← t05 − t06   < 2
        //     const storage t08 = fd.template add<0>(Y1, Z1);                          // t08 ← Y1 + Z1     < 4
        //     const storage t09 = fd.template add<0>(Y2, Z2);                          // t09 ← Y2 + Z2     < 4
        //     const storage t10 = fd.template mul<2>(t08, t09);                        // t10 ← t08 · t09   < 3
        //     const storage t11 = fd.template add<0>(t01, t02);                        // t11 ← t01 + t02   < 4
        //     const storage t12 = fd.template reduce<2>(fd.template sub<4>(t10, t11)); // t12 ← t10 − t11   < 2
        //     const storage t13 = fd.template add<0>(X1, Z1);                          // t13 ← X1 + Z1     < 4
        //     const storage t14 = fd.template add<0>(X2, Z2);                          // t14 ← X2 + Z2     < 4
        //     const storage t15 = fd.template mul<2>(t13, t14);                        // t15 ← t13 · t14   < 3
        //     const storage t16 = fd.template add<0>(t00, t02);                        // t16 ← t00 + t02   < 4
        //     const storage t17 = fd.template reduce<2>(fd.template sub<4>(t15, t16)); // t17 ← t15 − t16   < 2
        //     const storage t18 = fd.template dbl<2>(t00);                             // t18 ← t00 + t00   < 2
        //     const storage t19 = fd.template add<2>(t18, t00);                        // t19 ← t18 + t00   < 2
        //     const storage t20 = fd.template mul<2>(3 * B_VALUE, t02);                // t20 ← b3 · t02    < 2
        //     const storage t21 = fd.template add<2>(t01, t20);                        // t21 ← t01 + t20   < 2
        //     const storage t22 = fd.template sub<2>(t01, t20);                        // t22 ← t01 − t20   < 2
        //     const storage t23 = fd.template mul<2>(3 * B_VALUE, t17);                // t23 ← b3 · t17    < 2
        // #ifdef __CUDA_ARCH__
        //     // X3 ← t07 · t22 - t12 · t23
        //     const storage_wide t24_wide = fd.template mul_wide<0>(t12, t23);         //                   < 4*mod^2
        //     const storage_wide t25_wide = fd.template mul_wide<0>(t07, t22);         //                   < 4*mod^2
        //     storage_wide t25mt24_wide = fd.template sub_wide<4>(t25_wide, t24_wide); //                   < 4*mod^2
        //     fd.redc_wide_inplace(t25mt24_wide);                                      //                   < 2*mod, hi limbs 0
        //     const storage X3 = t25mt24_wide.get_lo();                                //                   < 2*mod
        //     // Y3 ← t22 · t21 + t23 · t19
        //     const storage t21_red = fd.template reduce<1>(t21);                      //                   < 1*mod
        //     const storage t19_red = fd.template reduce<1>(t19);                      //                   < 1*mod
        //     const storage_wide t27_wide = fd.template mul_wide<0>(t23, t19_red);     //                   < 2*mod^2
        //     const storage_wide t28_wide = fd.template mul_wide<0>(t22, t21_red);     //                   < 2*mod^2
        //     storage_wide t28pt27_wide = fd.template add_wide<4>(t28_wide, t27_wide); //                   < 4*mod^2
        //     fd.redc_wide_inplace(t28pt27_wide);                                      //                   < 2*mod, hi limbs 0
        //     const storage Y3 = t28pt27_wide.get_lo();                                //                   < 2*mod
        //     // Z3 ← t21 · t12 + t19 · t07
        //     const storage_wide t30_wide = fd.template mul_wide<0>(t19_red, t07);     //                   < 2*mod^2
        //     const storage_wide t31_wide = fd.template mul_wide<0>(t21_red, t12);     //                   < 2*mod^2
        //     storage_wide t31pt30_wide = fd.template add_wide<4>(t31_wide, t30_wide); //                   < 4*mod^2
        //     fd.redc_wide_inplace(t31pt30_wide);                                      //                   < 2*mod, hi limbs 0
        //     const storage Z3 = t31pt30_wide.get_lo();                                //                   < 2*mod
        // #else
        //     const storage t24 = fd.template mul<0>(t12, t23); // t24 ← t12 · t23   < 2
        //     const storage t25 = fd.template mul<0>(t07, t22); // t25 ← t07 · t22   < 2
        //     const storage X3 = fd.template sub<2>(t25, t24);  // X3 ← t25 − t24    < 2
        //     const storage t27 = fd.template mul<0>(t23, t19); // t27 ← t23 · t19   < 2
        //     const storage t28 = fd.template mul<0>(t22, t21); // t28 ← t22 · t21   < 2
        //     const storage Y3 = fd.template add<2>(t28, t27);  // Y3 ← t28 + t27    < 2
        //     const storage t30 = fd.template mul<0>(t19, t07); // t30 ← t19 · t07   < 2
        //     const storage t31 = fd.template mul<0>(t21, t12); // t31 ← t21 · t12   < 2
        //     const storage Z3 = fd.template add<2>(t31, t30);  // Z3 ← t31 + t30    < 2
        // #endif
        //     return {X3, Y3, Z3};
        // }

        // // https://eprint.iacr.org/2015/1060.pdf
        // static  __device__ __forceinline__ PointProjective add(const PointProjective &p1, const PointAffine &p2, const FD &fd) {
        //     const storage X1 = p1.x;
        //     const storage Y1 = p1.y;
        //     const storage Z1 = p1.z;
        //     const storage X2 = p2.x;
        //     const storage Y2 = p2.y;
        //     storage t0 = fd.template mul<0>(X1, X2);          // 1. t0 ← X1 · X2
        //     storage t1 = fd.template mul<0>(Y1, Y2);          // 2. t1 ← Y1 · Y2
        //     storage t3 = fd.template add<2>(X2, Y2);          // 3. t3 ← X2 + Y2
        //     storage t4 = fd.template add<2>(X1, Y1);          // 4. t4 ← X1 + Y1
        //     t3 = fd.template mul<0>(t3, t4);                  // 5. t3 ← t3 · t4
        //     t4 = fd.template add<2>(t0, t1);                  // 6. t4 ← t0 + t1
        //     t3 = fd.template sub<2>(t3, t4);                  // 7. t3 ← t3 − t4
        //     t4 = fd.template mul<0>(Y2, Z1);                  // 8. t4 ← Y2 · Z1
        //     t4 = fd.template add<2>(t4, Y1);                  // 9. t4 ← t4 + Y1
        //     storage Y3 = fd.template mul<0>(X2, Z1);          // 10. Y3 ← X2 · Z1
        //     Y3 = fd.template add<2>(Y3, X1);                  // 11. Y3 ← Y3 + X1
        //     storage X3 = fd.template dbl<2>(t0);              // 12. X3 ← t0 + t0
        //     t0 = fd.template add<2>(X3, t0);                  // 13. t0 ← X3 + t0
        //     storage t2 = fd.template mul<2>(3 * B_VALUE, Z1); // 14. t2 ← b3 · Z1
        //     storage Z3 = fd.template add<2>(t1, t2);          // 15. Z3 ← t1 + t2
        //     t1 = fd.template sub<2>(t1, t2);                  // 16. t1 ← t1 − t2
        //     Y3 = fd.template mul<2>(3 * B_VALUE, Y3);         // 17. Y3 ← b3 · Y3
        //     X3 = fd.template mul<0>(t4, Y3);                  // 18. X3 ← t4 · Y3
        //     t2 = fd.template mul<0>(t3, t1);                  // 19. t2 ← t3 · t1
        //     X3 = fd.template sub<2>(t2, X3);                  // 20. X3 ← t2 − X3
        //     Y3 = fd.template mul<0>(Y3, t0);                  // 21. Y3 ← Y3 · t0
        //     t1 = fd.template mul<0>(t1, Z3);                  // 22. t1 ← t1 · Z3
        //     Y3 = fd.template add<2>(t1, Y3);                  // 23. Y3 ← t1 + Y3
        //     t0 = fd.template mul<0>(t0, t3);                  // 24. t0 ← t0 · t3
        //     Z3 = fd.template mul<0>(Z3, t4);                  // 25. Z3 ← Z3 · t4
        //     Z3 = fd.template add<2>(Z3, t0);                  // 26. Z3 ← Z3 + t0
        //     return {X3, Y3, Z3};
        // }

        // // https://eprint.iacr.org/2015/1060.pdf
        // static  __device__ __forceinline__ PointProjective dbl(const PointProjective &point, const FD &fd) {
        //     const storage X = point.x;
        //     const storage Y = point.y;
        //     const storage Z = point.z;
        //     storage t0 = fd.template sqr<0>(Y);       // 1. t0 ← Y · Y
        //     storage Z3 = fd.template dbl<2>(t0);      // 2. Z3 ← t0 + t0
        //     Z3 = fd.template dbl<2>(Z3);              // 3. Z3 ← Z3 + Z3
        //     Z3 = fd.template dbl<2>(Z3);              // 4. Z3 ← Z3 + Z3
        //     storage t1 = fd.template mul<0>(Y, Z);    // 5. t1 ← Y · Z
        //     storage t2 = fd.template sqr<0>(Z);       // 6. t2 ← Z · Z
        //     t2 = fd.template mul<2>(3 * B_VALUE, t2); // 7. t2 ← b3 · t2
        //     storage X3 = fd.template mul<0>(t2, Z3);  // 8. X3 ← t2 · Z3
        //     storage Y3 = fd.template add<2>(t0, t2);  // 9. Y3 ← t0 + t2
        //     Z3 = fd.template mul<0>(t1, Z3);          // 10. Z3 ← t1 · Z3
        //     t1 = fd.template dbl<2>(t2);              // 11. t1 ← t2 + t2
        //     t2 = fd.template add<2>(t1, t2);          // 12. t2 ← t1 + t2
        //     t0 = fd.template sub<2>(t0, t2);          // 13. t0 ← t0 − t2
        //     Y3 = fd.template mul<0>(t0, Y3);          // 14. Y3 ← t0 · Y3
        //     Y3 = fd.template add<2>(X3, Y3);          // 15. Y3 ← X3 + Y3
        //     t1 = fd.template mul<0>(X, Y);            // 16. t1 ← X · Y
        //     X3 = fd.template mul<0>(t0, t1);          // 17. X3 ← t0 · t1
        //     X3 = fd.template dbl<2>(X3);              // 18. X3 ← X3 + X3
        //     return {X3, Y3, Z3};
        // }

        // static  __device__ __forceinline__ PointProjective mul(const unsigned scalar, const PointProjective &point, const FD &fd) {
        //     PointProjective result = PointProjective::point_at_infinity();
        //     PointProjective temp = point;
        //     unsigned l = scalar;
        //     bool is_zero = true;
        // #ifdef __CUDA_ARCH__
        // #pragma unroll
        // #endif
        //     for (unsigned i = 0; i < 32; i++) {
        //     if (l & 1) {
        //         result = is_zero ? temp : add(result, temp);
        //         is_zero = false;
        //     }
        //     l >>= 1;
        //     if (l == 0)
        //         break;
        //     temp = dbl(temp);
        //     }
        //     return result;
        // }


        // template <class FD_SCALAR, bool CHECK_ZERO = true>
        // static  __device__ __forceinline__ PointXYZZ mul(const typename FD_SCALAR::storage &scalar, const PointXYZZ &point, const FD &fd) {
        //     if (CHECK_ZERO) {
        //     if (unlikely(fd.is_zero(point.zz)))
        //         return PointXYZZ::point_at_infinity(fd);
        //     }
        //     PointXYZZ result = PointXYZZ::point_at_infinity(fd);
        //     unsigned count = FD_SCALAR::TLC;
        //     while (count != 0 && scalar.limbs[count - 1] == 0)
        //     count--;
        //     PointXYZZ temp = point;
        //     bool is_zero = true;
        //     for (unsigned i = 0; i < count; i++) {
        //     uint32_t limb = scalar.limbs[i];
        // #ifdef __CUDA_ARCH__
        // #pragma unroll
        // #endif
        //     for (unsigned j = 0; j < 32; j++) {
        //         if (limb & 1) {
        //         result = is_zero ? temp : add<false>(result, temp, fd);
        //         is_zero = false;
        //         }
        //         limb >>= 1;
        //         if (i == count - 1 && limb == 0)
        //         break;
        //         temp = dbl<false>(temp, fd);
        //     }
        //     }
        //     return result;
        // }

        // static  __device__ __forceinline__ PointProjective sub(const PointProjective &p1, const PointAffine &p2, const FD &fd) {
        //     return add(p1, PointAffine::neg(p2, fd), fd);
        // }

        // static  __device__ __forceinline__ PointProjective sub(const PointProjective &p1, const PointProjective &p2, const FD &fd) {
        //     return add(p1, PointProjective::neg(p2, fd), fd);
        // }

        // template <bool CHECK_ZERO = true, bool CHECK_DOUBLE = true>
        // static  __device__ __forceinline__ PointJacobian sub(const PointJacobian &p1, const PointAffine &p2, const FD &fd) {
        //     return add<CHECK_ZERO, CHECK_DOUBLE>(p1, PointAffine::neg(p2, fd), fd);
        // }

        // template <bool CHECK_ZERO = true, bool CHECK_DOUBLE = true>
        // static  __device__ __forceinline__ PointJacobian sub(const PointJacobian &p1, const PointJacobian &p2, const FD &fd) {
        //     return add<CHECK_ZERO, CHECK_DOUBLE>(p1, PointJacobian::neg(p2, fd), fd);
        // }

        // template <bool CHECK_ZERO = true, bool CHECK_DOUBLE = true>
        // static  __device__ __forceinline__ PointXYZZ sub(const PointXYZZ &p1, const PointAffine &p2, const FD &fd) {
        //     return add<CHECK_ZERO, CHECK_DOUBLE>(p1, PointAffine::neg(p2, fd), fd);
        // }

        // template <bool CHECK_ZERO = true, bool CHECK_DOUBLE = true>
        // static  __device__ __forceinline__ PointXYZZ sub(const PointXYZZ &p1, const PointXYZZ &p2, const FD &fd) {
        //     return add<CHECK_ZERO, CHECK_DOUBLE>(p1, PointXYZZ::neg(p2, fd), fd);
        // }
    };
}