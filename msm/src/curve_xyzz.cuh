#pragma once

#include "../../mont/src/field.cuh"
#include <iostream>

// #include "../../mont/src/bn254_scalar.cuh"
// #include "../../mont/src/bls12381_fq.cuh"
#include "../../mont/src/mnt4753_fq.cuh"

#ifdef __CUDA_ARCH__
#define likely(x) (__builtin_expect((x), 1))
#define unlikely(x) (__builtin_expect((x), 0))
#else
#define likely(x) (x) [[likely]]
#define unlikely(x) (x) [[unlikely]]
#endif 

#define TPI 4

namespace curve
{
    using mont::u32;
    using mont::usize;
    // using Params1 = bn254_scalar::Params;
    // using Params1 = bls12381_fq::Params;
    using Params1 = mnt4753_fq::Params;

    template <class Params>
    struct EC {
        template <u32 LIMBS_>
        struct PointXYZZ;
        template <u32 LIMBS_>
        struct PointAffine;

        // we assume that no points in pointaffine are identity
        template <u32 LIMBS_>
        struct PointAffine {
            static const usize N_WORDS = 2 * LIMBS_;

            using Element = mont::Element<Params1, LIMBS_>;

            Element x, y;

            friend std::ostream& operator<<(std::ostream &os, const PointAffine<LIMBS_> &p) {
                os << "{\n";
                os << "  .x = " << p.x << ",\n";
                os << "  .y = " << p.y << ",\n";
                // os << "  .x = ";
                // os << "0x";
                // for (usize i = LIMBS_ - 1; i >= 1; i--)
                //     os << std::hex << std::setfill('0') << std::setw(8) << p.x.n._limbs[i] << '_';
                // os << std::hex << std::setfill('0') << std::setw(8) << p.x.n._limbs[0];
                // os << ",\n";
                // os << "  .y = ";
                // os << "0x";
                // for (usize i = LIMBS_ - 1; i >= 1; i--)
                //     os << std::hex << std::setfill('0') << std::setw(8) << p.y.n._limbs[i] << '_';
                // os << std::hex << std::setfill('0') << std::setw(8) << p.y.n._limbs[0];
                // os << ",\n";
                os << "}";
                return os;
            }

            friend std::istream& operator>>(std::istream &is, PointAffine<LIMBS_> &p) {
                is >> std::hex;
                // char _;
                // is >> _ >> _;
                // for (int i = LIMBS_ - 1; i >= 1; i--)
                //     is >> p.x.n._limbs[i] >> _;
                // is >> p.x.n._limbs[0];
                // is >> _ >> _;
                // for (int i = LIMBS_ - 1; i >= 1; i--)
                //     is >> p.y.n._limbs[i] >> _;
                // is >> p.y.n._limbs[0];
                // return is;
                is >> p.x >> p.y;
                return is;
            }

            __host__ __device__ __forceinline__ PointAffine() {}
            __host__ __device__ __forceinline__ PointAffine(Element x, Element y) : x(x), y(y) {}

            __device__ __host__ __forceinline__ PointAffine<LIMBS_> neg() const & {
                return PointAffine<LIMBS_>(x, y.neg());
            }

            static __host__ __device__ __forceinline__ PointAffine<LIMBS_> load(const u32 *p) {
                auto x = Element::load(p);
                auto y = Element::load(p + LIMBS_);
                return PointAffine<LIMBS_>(x, y);
            }
            __host__ __device__ __forceinline__ void store(u32 *p) {
                x.store(p);
                y.store(p + LIMBS_);
            }

            // __device__ __forceinline__ void load_cg(const u32 *p) {
            //     int group_thread = threadIdx.x & (TPI-1);
            //     int PER_LIMBS = (Element::LIMBS + TPI - 1) / TPI;
            //     for(int i=group_thread*PER_LIMBS; i<(group_thread+1)*PER_LIMBS && i<Element::LIMBS; ++i) {
            //         x.n.limbs[i] = p[i];
            //         y.n.limbs[i] = p[Element::LIMBS + i];
            //     }
            // }
            // __device__ __forceinline__ void store_cg(u32 *p) {
            //     int group_thread = threadIdx.x & (TPI-1);
            //     int PER_LIMBS = (Element::LIMBS + TPI - 1) / TPI;
            //     for(int i=group_thread*PER_LIMBS; i<(group_thread+1)*PER_LIMBS && i<Element::LIMBS; ++i) {
            //         p[i] = x.n.limbs[i];
            //         p[Element::LIMBS + i] = y.n.limbs[i];
            //     }
            // }

            __host__ __device__ __forceinline__
                PointAffine<LIMBS_>
                operator=(const PointAffine<LIMBS_> &rhs) &
            {
                if(this != &rhs) {
                    x = rhs.x;
                    y = rhs.y;
                }
                return *this;
            }

            static __device__ __host__ __forceinline__ PointAffine<LIMBS_> identity() {
                return PointAffine<LIMBS_>(Element::zero(), Element::zero());
            }

            __device__ __host__ __forceinline__ bool is_identity() const & {
                return y.is_zero();
            }

            __device__ __host__ __forceinline__ bool operator==(const PointAffine<LIMBS_> &rhs) const & {
                return x == rhs.x && y == rhs.y;
            }

            // __device__ __host__ __forceinline__ bool is_on_curve() const & {
            //     Element t0, t1;
            //     u32 group_thread = threadIdx.x & (TPI-1);
            //     t0 = x.square();
            //     if (!Params::a_is_zero()) t0 = t0 + Params::template a<LIMBS_>(group_thread*LIMBS_, (group_thread+1)*LIMBS_);
            //     t0 = t0 * x;
            //     t0 = t0 + Params::template b<LIMBS_>(group_thread*LIMBS_, (group_thread+1)*LIMBS_);
            //     t1 = y.square();
            //     t0 = t1 - t0;
            //     return t0.is_zero();
            // }

            __device__ __host__ __forceinline__ PointXYZZ<LIMBS_> to_point() const& {
                if unlikely(is_identity()) return PointXYZZ<LIMBS_>::identity();
                return PointXYZZ<LIMBS_>(x, y, Element::one(), Element::one());
            }

            // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#doubling-mdbl-2008-s-1
            __device__ __host__ __forceinline__ PointXYZZ<LIMBS_> add_self() const& {
#ifdef __CUDA_ARCH__
                u32 group_thread = threadIdx.x & (TPI-1);
                auto u = y + y;
                auto v = u.square();
                auto w = u * v;
                auto s = x * v;
                auto x2 = x.square();
                auto m = x2 + x2 + x2;
                if (!Params::a_is_zero()) 
                    m = m + Params::template a<LIMBS_>(group_thread*LIMBS_, (group_thread+1)*LIMBS_);
                auto x3 = m.square() - s - s;
                auto y3 = m * (s - x3) - w * y;
                return PointXYZZ<LIMBS_>(x3, y3, v, w);
#else
                auto u = y + y;
                auto v = u.square();
                auto w = u * v;
                auto s = x * v;
                auto x2 = x.square();
                auto m = x2 + x2 + x2;
                if (!Params::a_is_zero()) 
                    m = m + Params::template a<LIMBS_>(0, LIMBS_);
                auto x3 = m.square() - s - s;
                auto y3 = m * (s - x3) - w * y;
                return PointXYZZ(x3, y3, v, w);
#endif
            }

            __host__ __device__ void device_print() const &
            {
                printf(
                    "{ x = %x %x %x %x %x %x %x %x\n, y = %x %x %x %x %x %x %x %x}\n",
                    x.n._limbs[7], x.n._limbs[6], x.n._limbs[5], x.n._limbs[4], x.n._limbs[3], x.n._limbs[2], x.n._limbs[1], x.n._limbs[0], 
                    y.n._limbs[7], y.n._limbs[6], y.n._limbs[5], y.n._limbs[4], y.n._limbs[3], y.n._limbs[2], y.n._limbs[1], y.n._limbs[0]
                );
            }

            __device__ __host__ __forceinline__ PointAffine<LIMBS_> shuffle_down(const u32 delta, u32 mask = 0xFFFFFFFF) const & {
                PointAffine<LIMBS_> res;
                #pragma unroll
                for (usize i = 0; i < LIMBS_; i++) {
                    res.x.n._limbs[i] = __shfl_down_sync(mask, x.n._limbs[i], delta);
                    res.y.n._limbs[i] = __shfl_down_sync(mask, y.n._limbs[i], delta);
                }
                return res;
            }
         };


        //  https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html
        //  x=X/ZZ
        //  y=Y/ZZZ
        //  ZZ^3=ZZZ^2
        template <u32 LIMBS_>
        struct PointXYZZ {
            static const usize N_WORDS = 4 * LIMBS_;
            using Element = mont::Element<Params1, LIMBS_>;
            Element x, y, zz, zzz;

            __host__ __device__ __forceinline__ PointXYZZ() {};
            __host__ __device__ __forceinline__ PointXYZZ(Element x, Element y, Element zz, Element zzz) : x(x), y(y), zz(zz), zzz(zzz) {}

            static __host__ __device__ __forceinline__ PointXYZZ<LIMBS_> load(const u32 *p) {
                auto x = Element::load(p);
                auto y = Element::load(p + LIMBS_);
                auto zz = Element::load(p + LIMBS_ * 2);
                auto zzz = Element::load(p + LIMBS_ * 3);
                return PointXYZZ<LIMBS_>(x, y, zz, zzz);
            }
            __host__ __device__ __forceinline__ void store(u32 *p) {
                x.store(p);
                y.store(p + LIMBS_);
                zz.store(p + LIMBS_ * 2);
                zzz.store(p + LIMBS_ * 3);
            }
            __host__ __device__ __forceinline__
                PointXYZZ<LIMBS_>
                operator=(const PointXYZZ<LIMBS_> &rhs) &
            {
                if(this != &rhs) {
                    x = rhs.x;
                    y = rhs.y;
                    zz = rhs.zz;
                    zzz = rhs.zzz;
                }
                return *this;
            }

            static constexpr __device__ __host__ __forceinline__ PointXYZZ<LIMBS_> identity() {
                return PointXYZZ<LIMBS_>(Element::zero(), Element::zero(), Element::zero(), Element::one());
            }

            __device__ __host__ __forceinline__ bool is_identity() const & {
                return zz.is_zero();
            }

            __device__ __host__ __forceinline__ PointXYZZ<LIMBS_> neg() const & {
                return PointXYZZ<LIMBS_>(x, y.neg(), zz, zzz);
            }

            __host__ __device__ __forceinline__ bool operator==(const PointXYZZ<LIMBS_> &rhs) const & {
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
            // __host__ __device__ __forceinline__ bool is_on_curve() const & {
            //     // auto self = normalized();
            //     u32 group_thread = threadIdx.x & (TPI-1);
            //     auto y2 = y.square();
            //     auto x3 = x.square() * x;
            //     auto zz2 = zz.square();
            //     auto zz3 = zz * zz2;
            //     auto zzz2 = zzz.square();
            //     if (zz3 != zzz2) return false;
            //     Element a_x_zz2;
            //     if (Params::a_is_zero()) a_x_zz2 = Element::zero();
            //     else a_x_zz2 = Params::template a<LIMBS_>(group_thread*LIMBS_, (group_thread+1)*LIMBS_) * x * zz2;
            //     auto b_zz3 = Params::template b<LIMBS_>(group_thread*LIMBS_, (group_thread+1)*LIMBS_) * zz3;
            //     return y2 == x3 + a_x_zz2 + b_zz3;
            // }

            __device__ __host__ __forceinline__ bool is_elements_lt_2m() const &
            {
                return x.lt_2m() && y.lt_2m() && zz.lt_2m() && zzz.lt_2m();
            }

            // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#scaling-z
            __device__ __host__ __forceinline__ PointAffine<LIMBS_> to_affine() const & {
                // auto self = normalized();
                auto A = zzz.invert();
                auto B = (zz * A).square();
                auto X3 = x * B;
                auto Y3 = y * A;
                return PointAffine<LIMBS_>(X3, Y3);
            }

            __device__ __host__ __forceinline__ PointXYZZ<LIMBS_> add(const PointXYZZ<LIMBS_> &rhs) const &
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
                return PointXYZZ<LIMBS_>(x3, y3, zz3, zzz3);
            }

            __host__ __device__ void device_print() const &
            {
                printf(
                    "{ x = %x %x %x %x %x %x %x %x\n, y = %x %x %x %x %x %x %x %x\n, zz = %x %x %x %x %x %x %x %x\n, zzz = %x %x %x %x %x %x %x %x }\n",
                    x.n._limbs[7], x.n._limbs[6], x.n._limbs[5], x.n._limbs[4], x.n._limbs[3], x.n._limbs[2], x.n._limbs[1], x.n._limbs[0], 
                    y.n._limbs[7], y.n._limbs[6], y.n._limbs[5], y.n._limbs[4], y.n._limbs[3], y.n._limbs[2], y.n._limbs[1], y.n._limbs[0], 
                    zz.n._limbs[7], zz.n._limbs[6], zz.n._limbs[5], zz.n._limbs[4], zz.n._limbs[3], zz.n._limbs[2], zz.n._limbs[1], zz.n._limbs[0], 
                    zzz.n._limbs[7], zzz.n._limbs[6], zzz.n._limbs[5], zzz.n._limbs[4], zzz.n._limbs[3], zzz.n._limbs[2], zzz.n._limbs[1], zzz.n._limbs[0]
                );
            }

            // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-add-2008-s
            __device__ __host__ __forceinline__ PointXYZZ<LIMBS_> operator + (const PointXYZZ<LIMBS_> &rhs) const & {
                return add(rhs);
            }

            __device__ __host__ __forceinline__ PointXYZZ<LIMBS_> operator - (const PointXYZZ<LIMBS_> &rhs) const & {
                return *this + rhs.neg();
            }

            // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-madd-2008-s
            __device__ __host__ __forceinline__ PointXYZZ<LIMBS_> add(const PointAffine<LIMBS_> &rhs) const & {
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
                return PointXYZZ<LIMBS_>(x3, y3, zz3, zzz3);
            }

            __device__ __host__ __forceinline__ PointXYZZ<LIMBS_> operator + (const PointAffine<LIMBS_> &rhs) const & {
                return add(rhs);
            }

            __device__ __host__ __forceinline__ PointXYZZ<LIMBS_> operator - (const PointAffine<LIMBS_> &rhs) const & {
                return *this + rhs.neg();
            }

            // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#doubling-dbl-2008-s-1
            __device__ __host__ __forceinline__ PointXYZZ<LIMBS_> self_add() const & {
#ifdef __CUDA_ARCH__
                u32 group_thread = threadIdx.x & (TPI-1);
                if unlikely(zz.is_zero()) return *this;
                auto u = y + y;
                auto v = u.square();
                auto w = u * v;
                auto s = x * v;
                auto x2 = x.square();
                auto m = x2 + x2 + x2;
                if (!Params::a_is_zero()) {
                    m = m + (Params::template a<LIMBS_>(group_thread*LIMBS_, (group_thread+1)*LIMBS_) * zz.square());
                }
                auto x3 = m.square() - s - s;
                auto y3 = m * (s - x3) - w * y;
                auto zz3 = v * zz;
                auto zzz3 = w * zzz;
                return PointXYZZ<LIMBS_>(x3, y3, zz3, zzz3);
#else
                if unlikely(zz.is_zero()) return *this;
                auto u = y + y;
                auto v = u.square();
                auto w = u * v;
                auto s = x * v;
                auto x2 = x.square();
                auto m = x2 + x2 + x2;
                if (!Params::a_is_zero()) 
                    m = m + (Params::template a<LIMBS_>(0, LIMBS_) * zz.square());
                auto x3 = m.square() - s - s;
                auto y3 = m * (s - x3) - w * y;
                auto zz3 = v * zz;
                auto zzz3 = w * zzz;
                return PointXYZZ(x3, y3, zz3, zzz3);
#endif
            }

            static __device__ __host__ __forceinline__ void multiple_iter(const PointXYZZ<LIMBS_> &p, bool &found_one, PointXYZZ<LIMBS_> &res, u32 n) {
                for (int i = 31; i >= 0; i--) {
                    if (found_one) res = res.self_add();
                    if ((n >> i) & 1) {
                        found_one = true;
                        res = res + p;
                    }
                }
            }

            __device__ __host__ __forceinline__ PointXYZZ<LIMBS_> multiple(u32 n) const & {
                auto res = identity();
                bool found_one = false;
                multiple_iter(*this, found_one, res, n);
                return res;
            }

            __device__ __host__ __forceinline__ PointXYZZ<LIMBS_> shuffle_down(const u32 delta, u32 mask = 0xFFFFFFFF) const & {
                PointXYZZ<LIMBS_> res;
                #pragma unroll
                for (usize i = 0; i < LIMBS_; i++) {
                    res.x.n._limbs[i] = __shfl_down_sync(mask, x.n._limbs[i], delta);
                    res.y.n._limbs[i] = __shfl_down_sync(mask, y.n._limbs[i], delta);
                    res.zz.n._limbs[i] = __shfl_down_sync(mask, zz.n._limbs[i], delta);
                    res.zzz.n._limbs[i] = __shfl_down_sync(mask, zzz.n._limbs[i], delta);
                }
                return res;
            }
        };
    };
}