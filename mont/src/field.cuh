#ifndef MONT_H
#define MONT_H

#include <iostream>
#include <iomanip>
#include <tuple>

#define BIG_INTEGER_CHUNKS8(c7, c6, c5, c4, c3, c2, c1, c0) {c0, c1, c2, c3, c4, c5, c6, c7}
#define BIG_INTEGER_CHUNKS16(c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0) \
  {c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15}

#include "ptx.cuh"

namespace mont
{

  using u32 = u_int32_t;
  using u64 = u_int64_t;
  using usize = size_t;

  namespace host_arith
  {
    __host__ u32 addc(u32 a, u32 b, u32 &carry)
    {
      u64 ret = (u64)a + (u64)b + (u64)carry;
      carry = (u32)(ret >> 32);
      return ret;
    }

    __host__ u32 subb(u32 a, u32 b, u32 &borrow)
    {
      u64 ret = (u64)a - (u64)b - (u64)(borrow >> 31);
      borrow = (u32)(ret >> 32);
      return ret;
    }

    __host__ u32 madc(u32 a, u32 b, u32 c, u32 &carry)
    {
      u64 ret = (u64)b * (u64)c + (u64)a + (u64)carry;
      carry = (u32)(ret >> 32);
      return ret;
    }

    template <usize N>
    __host__ u32 sub(u32 *r, const u32 *a, const u32 *b)
    {
      u32 carry = 0;
#pragma unroll
      for (usize i = 0; i < N; i++)
        r[i] = host_arith::subb(a[i], b[i], carry);
      return carry;
    }

    template <usize N>
    __host__ u32 add(u32 *r, const u32 *a, const u32 *b)
    {
      u32 carry = 0;
#pragma unroll
      for (usize i = 0; i < N; i++)
        r[i] = host_arith::addc(a[i], b[i], carry);
      return carry;
    }

    template <usize N>
    __host__ void sub_modulo(u32 *r, const u32 *a, const u32 *b, const u32 *m)
    {
      u32 borrow = sub<N>(r, a, b);
      if (borrow)
        add<N>(r, r, m);
    }

    template <usize N>
    __host__ void add_modulo(u32 *r, const u32 *a, const u32 *b, const u32 *m)
    {
      add<N>(r, a, b);
      sub_modulo<N>(r, r, m);
    }

    template <usize N>
    __host__ void multiply(u32 *r, const u32 *a, const u32 *b)
    {
      u32 carry = 0;

#pragma unroll
      for (usize j = 0; j < N; j++)
        r[j] = madc(0, a[0], b[j], carry);
      r[N] = carry;
      carry = 0;

#pragma unroll
      for (usize i = 1; i < N; i++)
      {
#pragma unroll
        for (usize j = 0; j < N; j++)
          r[i + j] = madc(r[i + j], a[i], b[j], carry);
        r[N + i] = carry;
        carry = 0;
      }
    }

    template <usize N>
    __host__ void montgomery_reduction(u32 *a, const u32 *m, const u32 m_prime)
    {
      u32 k, carry;
      u32 carry2 = 0;
#pragma unroll
      for (usize i = 0; i < N; i++)
      {
        k = a[i] * m_prime;
        carry = 0;
        madc(a[i], k, m[0], carry);
#pragma unroll
        for (usize j = 1; j < N; j++)
          a[i + j] = madc(a[i + j], k, m[j], carry);
        a[i + N] = addc(a[i + N], carry2, carry);
        carry2 = carry;
      }
    }

    template <usize N>
    __host__ void montgomery_multiplication(u32 *r, const u32 *a, const u32 *b, const u32 *m, const u32 m_prime)
    {
      u32 product[2 * N];
      multiply<N>(product, a, b);
      montgomery_reduction<N>(product, m, m_prime);
      memcpy(r, product + N, N * sizeof(u32));
      sub_modulo<N>(r, r, m, m);
    }
  }

  namespace device_arith
  {
    // Multiply even limbs of a big number `a` with u32 `b`, writing result to `r`.
    // `a` has `N` limbs, and so does `r`.
    //   | 0      | a2     | 0       | a0    |
    // *                             | b     |
    //   -------------------------------------
    //                     | a0 * b          |
    // + | a2 * b          |
    //   -------------------------------------
    //   | r                                 |
    template <usize N>
    __device__ __forceinline__ void
    multiply_n_1_even(u32 *r, const u32 *a, const u32 b)
    {
#pragma unroll
      for (usize i = 0; i < N; i += 2)
      {
        r[i] = ptx::mul_lo(a[i], b);
        r[i + 1] = ptx::mul_hi(a[i], b);
      }
    }

    // Multiply even limbs of big number `a` with u32 `b`, adding result to `c`, with an optional carry-in `carry_in`.
    // `a` has `N` limbs.
    // Final result written to `r`.
    // `CARRY_IN` controls whether to enable the parameter `carry_in`.
    // Both `a` and `r` has `N` limbs.
    //   | 0      | a2     | 0       | a0    |
    // *                             | b     |
    //   -------------------------------------
    //                     | a0 * b          |
    //   | a2 * b          |
    // + | c                                 |
    //   -------------------------------------
    //   | acc                               |
    template <usize N, bool CARRY_IN = false, bool CARRY_OUT = false>
    __device__ __forceinline__ u32 mad_n_1_even(u32 *acc, const u32 *a, const u32 b, const u32 *c, const u32 carry_in = 0)
    {
      if (CARRY_IN)
        ptx::add_cc(UINT32_MAX, carry_in);
      acc[0] = CARRY_IN ? ptx::madc_lo_cc(a[0], b, c[0]) : ptx::mad_lo_cc(a[0], b, c[0]);
      acc[1] = ptx::madc_hi_cc(a[0], b, c[1]);

#pragma unroll
      for (usize i = 2; i < N; i += 2)
      {
        acc[i] = ptx::madc_lo_cc(a[i], b, c[i]);
        acc[i + 1] = ptx::madc_hi_cc(a[i], b, c[i + 1]);
      }

      if (CARRY_OUT)
        return ptx::addc(0, 0);
      return 0;
    }

    // Like `mad_n_1_even`, but the result of multiplication is accumulated in `acc`.
    template <usize N, bool CARRY_IN = false, bool CARRY_OUT = false>
    __device__ __forceinline__
        u32
        mac_n_1_even(u32 *acc, const u32 *a, const u32 b, const u32 carry_in = 0)
    {
      return mad_n_1_even<N, CARRY_IN, CARRY_OUT>(acc, a, b, acc);
    }

    // Multiplies `a` and `b`, where `a` has `N` limbs.
    // The multiplication result is added to `c` and `d` in the following way.
    // `CARRY_IN` controls `carry_for_low`, and `CARRY_OUT` controls return value.
    //                   | a3    | 0     | a1    | 0    |
    // *                                 | b     |
    //   -----------------------------------------
    //                           | a1 * b        |
    //           | a3 * b        |
    //           | d     | c     | odd1  | odd0  |
    //                                   | cr_l  |        cr_l is `carry_for_low`
    // +         | cr_h  |                                cr_h is `carry_for_high`
    //   -----------------------------------------
    //   | cr    | odd                           |
    //
    //   `even` is same as `mad_n_1_even`
    template <usize N, bool CARRY_OUT = false, bool CARRY_IN = false>
    __device__ __forceinline__
        u32
        mad_row(
            u32 *odd,
            u32 *even,
            const u32 *a,
            const u32 b,
            const u32 c = 0,
            const u32 d = 0,
            const u32 carry_for_high = 0,
            const u32 carry_for_low = 0)
    {
      mac_n_1_even<N - 2, CARRY_IN>(odd, a + 1, b, carry_for_low);
      odd[N - 2] = ptx::madc_lo_cc(a[N - 1], b, c);
      odd[N - 1] = CARRY_OUT ? ptx::madc_hi_cc(a[N - 1], b, d) : ptx::madc_hi(a[N - 1], b, d);
      u32 cr = CARRY_OUT ? ptx::addc(0, 0) : 0;
      mac_n_1_even<N, false>(even, a, b);
      if (CARRY_OUT)
      {
        odd[N - 1] = ptx::addc_cc(odd[N - 1], carry_for_high);
        cr = ptx::addc(cr, 0);
      }
      else
        odd[N - 1] = ptx::addc(odd[N - 1], carry_for_high);
      return cr;
    }

    // Similar to `mad_row`, but with c, d set to zero
    template <usize N, bool CARRY_OUT = false, bool CARRY_IN = false>
    __device__ __forceinline__
        u32
        mac_row(
            u32 *odd,
            u32 *even,
            const u32 *a,
            const u32 b,
            const u32 carry_for_high = 0,
            const u32 carry_for_low = 0)
    {
      mac_n_1_even<N, CARRY_IN>(odd, a + 1, b, carry_for_low);
      mac_n_1_even<N, false>(even, a, b);
      u32 cr = 0;
      if (CARRY_OUT)
      {
        odd[N - 1] = ptx::addc_cc(odd[N - 1], carry_for_high);
        cr = ptx::addc(cr, 0);
      }
      else
        odd[N - 1] = ptx::addc(odd[N - 1], carry_for_high);
      return cr;
    }

    // Let `r` be `a` multiplied by `b`.
    // `a` and `b` both have `N` limbs and `r` has `2 * N` limbs.
    // Implements elementry school multiplication algorithm.
    template <usize N>
    __device__ __forceinline__ void multiply_naive(u32 *r, const u32 *a, const u32 *b)
    {
      u32 *even = r;

      __align__(16) u32 odd[2 * N - 2];
      multiply_n_1_even<N>(even, a, b[0]);
      multiply_n_1_even<N>(odd, a + 1, b[0]);
      mad_row<N>(&even[2], &odd[0], a, b[1]);

#pragma unroll
      for (usize i = 2; i < N - 1; i += 2)
      {
        mad_row<N>(&odd[i], &even[i], a, b[i]);
        mad_row<N>(&even[i + 2], &odd[i], a, b[i + 1]);
      }

      even[1] = ptx::add_cc(even[1], odd[0]);
      usize i;
#pragma unroll
      for (i = 1; i < 2 * N - 2; i++)
        even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
      even[i + 1] = ptx::addc(even[i + 1], 0);
    }

    // Let `r` be sum of `a` and `b`.
    // `a`, `b`, `r` all have `N` limbs.
    template <usize N>
    __device__ __forceinline__
        u32
        add(u32 *r, const u32 *a, const u32 *b)
    {
      r[0] = ptx::add_cc(a[0], b[0]);
#pragma unroll
      for (usize i = 1; i < N; i++)
        r[i] = ptx::addc_cc(a[i], b[i]);
      return ptx::addc(0, 0);
    }

    // Let `r` be difference of `a` and `b`.
    // `a`, `b`, `r` all have `N` limbs.
    template <usize N>
    __device__ __forceinline__
        u32
        sub(u32 *r, const u32 *a, const u32 *b)
    {
      r[0] = ptx::sub_cc(a[0], b[0]);
#pragma unroll
      for (usize i = 1; i < N; i++)
        r[i] = ptx::subc_cc(a[i], b[i]);
      return ptx::subc(0, 0);
    }

    // Multiplies `a` and `b`, adding result to `in1` and `in2`.
    // `a` and `b` have `N / 2` limbs, while `in1` and `in2` have `N` limbs.
    template <usize N>
    __device__ __forceinline__ void mad2_rows(u32 *r, const u32 *a, const u32 *b, const u32 *in1, const u32 *in2)
    {
      __align__(16) u32 odd[N - 2];
      u32 *even = r;
      u32 first_row_carry = mad_n_1_even<(N >> 1), false, true>(even, a, b[0], in1);
      u32 carry = mad_n_1_even<(N >> 1), false, true>(odd, &a[1], b[0], &in2[1]);

#pragma unroll
      for (usize i = 2; i < ((N >> 1) - 1); i += 2)
      {
        carry = mad_row<(N >> 1), true, false>(
            &even[i], &odd[i - 2], a, b[i - 1], in1[(N >> 1) + i - 2], in1[(N >> 1) + i - 1], carry);
        carry = mad_row<(N >> 1), true, false>(
            &odd[i], &even[i], a, b[i], in2[(N >> 1) + i - 1], in2[(N >> 1) + i], carry);
      }
      mad_row<(N >> 1), false, true>(
          &even[N >> 1], &odd[(N >> 1) - 2], a, b[(N >> 1) - 1], in1[N - 2], in1[N - 1], carry, first_row_carry);

      even[0] = ptx::add_cc(even[0], in2[0]);
      usize i;
#pragma unroll
      for (i = 0; i < N - 2; i++)
        even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
      even[i + 1] = ptx::addc(even[i + 1], in2[i + 1]);
    }

    // Compute `r = a * b` where `r` has `N * 2` limbs while `a` and `b` have `N` limbs.
    // Implements 1-layer Kruskaba Algorithm.
    template <usize N>
    __device__ __forceinline__ void multiply(u32 *r, const u32 *a, const u32 *b)
    {
      if (N > 2)
      {
        multiply_naive<(N >> 1)>(r, a, b);
        multiply_naive<(N >> 1)>(&r[N], &a[N >> 1], &b[N >> 1]);
        __align__(16) u32 middle_part[N];
        __align__(16) u32 diffs[N];
        u32 carry1 = sub<(N >> 1)>(diffs, &a[N >> 1], a);
        u32 carry2 = sub<(N >> 1)>(&diffs[N >> 1], b, &b[N >> 1]);
        mad2_rows<N>(middle_part, diffs, &diffs[N >> 1], r, &r[N]);
        if (carry1)
          sub<(N >> 1)>(&middle_part[N >> 1], &middle_part[N >> 1], &diffs[N >> 1]);
        if (carry2)
          sub<(N >> 1)>(&middle_part[N >> 1], &middle_part[N >> 1], diffs);
        u32 carry = add<N>(&r[N >> 1], &r[N >> 1], middle_part);

        r[N + (N >> 1)] = ptx::add_cc(r[N + (N >> 1)], carry);
#pragma unroll
        for (usize i = N + (N >> 1) + 1; i < 2 * N; i++)
          r[i] = ptx::addc_cc(r[i], 0);
      }
      else if (N == 2)
      {
        __align__(8) uint32_t odd[2];
        r[0] = ptx::mul_lo(a[0], b[0]);
        r[1] = ptx::mul_hi(a[0], b[0]);
        r[2] = ptx::mul_lo(a[1], b[1]);
        r[3] = ptx::mul_hi(a[1], b[1]);
        odd[0] = ptx::mul_lo(a[0], b[1]);
        odd[1] = ptx::mul_hi(a[0], b[1]);
        odd[0] = ptx::mad_lo(a[1], b[0], odd[0]);
        odd[1] = ptx::mad_hi(a[1], b[0], odd[1]);
        r[1] = ptx::add_cc(r[1], odd[0]);
        r[2] = ptx::addc_cc(r[2], odd[1]);
        r[3] = ptx::addc(r[3], 0);
      }
      else if (N == 1)
      {
        r[0] = ptx::mul_lo(a[0], b[0]);
        r[1] = ptx::mul_hi(a[0], b[0]);
      }
    }

    // Computes `r = x >> k` where `r` and `x` have `N` limbs.
    template <usize N>
    __host__ __device__ __forceinline__ void slr(u32 *r, const u32 *x, const u32 k)
    {
      if (k == 0)
      {
#pragma unroll
        for (usize i = 0; i < N; i++)
          r[i] = x[i];
        return;
      }
#pragma unroll
      for (usize i = 1; i <= N; i++)
      {
        if (k < i * 32)
        {

          u32 k_lo = k - (i - 1) * 32;
          u32 k_hi = i * 32 - k;
#pragma unroll
          for (int j = N - 1; j > N - i; j--)
            r[j] = 0;
          r[N - i] = x[N - 1] >> k_lo;
#pragma unroll
          for (int j = N - i - 1; j >= 0; j--)
            r[j] = (x[j + i] << k_hi) | (x[j + i - 1] >> k_lo);
          return;
        }
      }
    }

    // Apply Montgomery Reduction to `a`, with modulus `m` and `m_prime` satisfying
    //    m m_prime = -1 (mod 2^32)
    template <usize N>
    __device__ __forceinline__ void montgomery_reduction(u32 *a, const u32 *m, const u32 m_prime)
    {
#pragma unroll
      for (usize i = 0; i < N - 1; i += 1)
      {
        u32 u = a[i] * m_prime;
        mac_n_1_even<N, false, false>(&a[i], m, u);
        a[i + N] = ptx::addc(a[i + N], 0);
        mac_n_1_even<N, false, false>(&a[i + 1], &m[1], u);
        a[i + N + 1] = ptx::addc(a[i + N + 1], 0);
      }

      u32 u = a[N - 1] * m_prime;
      mac_n_1_even<N, false, false>(&a[N - 1], m, u);
      a[2 * N - 1] = ptx::addc(a[2 * N - 1], 0);
      mac_n_1_even<N, false, false>(&a[N], &m[1], u);
    }

    // Computes `r = a - b mod m`.
    // `r`, `a`, `b`, `m` all have limbs `N`
    template <usize N>
    __device__ __forceinline__ void sub_modulo(u32 *r, const u32 *a, const u32 *b, const u32 *m)
    {
      u32 borrow = sub<N>(r, a, b);
      if (borrow)
        add<N>(r, r, m);
    }

    // Computes `r = a + b mod m`.
    // `r`, `a`, `b`, `m` all have limbs `N`
    template <usize N>
    __device__ __forceinline__ void add_modulo(u32 *r, const u32 *a, const u32 *b, const u32 *m)
    {
      add<N>(r, a, b);
      sub_modulo<N>(r, r, m, m);
    }

    // Computes `r = a * b mod m`.
    // `r`, `a`, `b`, `m` all have limbs `N`
    template <usize N>
    __device__ __forceinline__ void montgomery_multiplication(u32 *r, const u32 *a, const u32 *b, const u32 *m, const u32 m_prime)
    {
      __align__(16) u32 prod[2 * N];
      multiply_naive<N>(prod, a, b);
      montgomery_reduction<N>(prod, m, m_prime);
#pragma unroll
      for (usize i = N; i < 2 * N; i++)
        r[i - N] = prod[i];
      sub_modulo<N>(r, r, m, m);
    }
  }

  template <usize LIMBS>
  struct
      __align__(16)
          Number
  {
    u32 limbs[LIMBS];

    // Host-Device methods

    __device__ __host__
    Number<LIMBS>() {}

    static __device__ __host__ __forceinline__
        Number<LIMBS>
        two()
    {
      Number<LIMBS> r;
      for (usize i = 1; i < LIMBS; i++)
        r.limbs[i] = 0;
      r.limbs[0] = 2;
      return r;
    }

    // Contract: `limbs.size() <= LIMBS`
    __device__ __host__
    Number<LIMBS>(std::initializer_list<u32> limbs)
    {
      auto iter = limbs.begin();
      for (int i = limbs.size() - 1; i >= 0; i--)
      {
        this->limbs[i] = *iter;
        iter++;
      }
      for (int i = limbs.size(); i < LIMBS; i++)
        this->limbs[i] = 0;
    }

    static __device__ __host__ __forceinline__
        Number<LIMBS>
        load(const u32 *p)
    {
      Number<LIMBS> r;
#pragma unroll
      for (usize i = 0; i < LIMBS; i++)
        r.limbs[i] = p[i];
      return r;
    }

    __device__ __host__ __forceinline__ void store(u32 * p) const &
    {
#pragma unroll
      for (usize i = 0; i < LIMBS; i++)
        p[i] = limbs[i];
    }

    static __device__ __host__ __forceinline__
        Number<LIMBS>
        zero()
    {
      Number<LIMBS> r;
      memset(r.limbs, 0, LIMBS * sizeof(u32));
      return r;
    }

    __host__ __device__ __forceinline__
        Number<LIMBS>
        slr(u32 k) const &
    {
      Number<LIMBS> r;
      device_arith::slr<LIMBS>(r.limbs, limbs, k);
      return r;
    }

    __host__ __device__ __forceinline__
        u32
        bit_slice(u32 lo, u32 n_bits)
    {
      Number<LIMBS> t = slr(lo);
      return t.limbs & ~((u32)0 - (1 << n_bits));
    }

    __host__ __device__ __forceinline__ bool
    operator==(const Number<LIMBS> &rhs) const &
    {
      bool r = true;
#pragma unroll
      for (usize i = 0; i < LIMBS; i++)
        r = r && (limbs[i] == rhs.limbs[i]);
      return r;
    }

    __host__ __device__ __forceinline__ bool
    operator!=(const Number<LIMBS> &rhs) const &
    {
      bool r = false;
#pragma unroll
      for (usize i = 0; i < LIMBS; i++)
        r = r || (limbs[i] != rhs.limbs[i]);
      return r;
    }

    __host__ __device__ __forceinline__ bool
    is_zero() const &
    {
      bool r = true;
#pragma unroll
      for (usize i = 0; i < LIMBS; i++)
        r = r && (limbs[i] == 0);
      return r;
    }

    // Device methods

    __device__ __forceinline__
        Number<LIMBS * 2>
        operator*(const Number &rhs) const &
    {
      Number<LIMBS * 2> r;
      device_arith::multiply_naive<LIMBS>(r.limbs, limbs, rhs.limbs);
      return r;
    }

    __device__ __forceinline__
        Number<LIMBS>
        operator+(const Number &rhs) const &
    {
      Number<LIMBS> r;
      device_arith::add<LIMBS>(r.limbs, limbs, rhs.limbs);
      return r;
    }

    __device__ __forceinline__
        Number<LIMBS>
        operator-(const Number &rhs) const &
    {
      Number<LIMBS> r;
      device_arith::sub<LIMBS>(r.limbs, limbs, rhs.limbs);
      return r;
    }

    __device__ __forceinline__
        Number<LIMBS>
        sub_borrowed(const Number &rhs, u32 &borrow_ret) const &
    {
      Number<LIMBS> r;
      borrow_ret = device_arith::sub<LIMBS>(r.limbs, limbs, rhs.limbs);
      return r;
    }

    __device__ __forceinline__
        Number<LIMBS * 2>
        square() const &
    {
      return *this * *this;
    }

    // Host methods

    __host__
        Number<LIMBS>
        host_add(const Number<LIMBS> &rhs) const &
    {
      Number<LIMBS> r;
      host_arith::add<LIMBS>(r.limbs, limbs, rhs.limbs);
      return r;
    }

    __host__
        Number<LIMBS>
        host_sub_borrowed(const Number<LIMBS> &rhs, u32 &borrow_ret) const &
    {
      borrow_ret = 0;
      Number<LIMBS> r;
      borrow_ret = host_arith::sub<LIMBS>(r.limbs, limbs, rhs.limbs);
      return r;
    }

    __host__
        Number<LIMBS>
        host_sub(const Number<LIMBS> &rhs) const &
    {
      u32 useless;
      return host_sub_borrowed(rhs, useless);
    }

    __host__
        Number<LIMBS * 2>
        host_mul(const Number<LIMBS> &rhs) const &
    {
      Number<LIMBS * 2> r;
      host_arith::multiply<LIMBS>(r.limbs, limbs, rhs.limbs);
      return r;
    }

    __host__
        Number<LIMBS * 2>
        host_square() const &
    {
      return host_mul(*this);
    }
  };

  template <usize LIMBS>
  struct Element
  {
    Number<LIMBS> n;

    __host__ __device__ Element() {}

    static __host__ __device__ __forceinline__
        Element<LIMBS>
        load(const u32 *p)
    {
      Element<LIMBS> r;
      r.n = Number<LIMBS>::load(p);
      return r;
    }

    __host__ __device__ __forceinline__ void store(u32 *p)
    {
      n.store(p);
    }

    static __host__ __device__
        Element<LIMBS>
        zero()
    {
      Element<LIMBS> r;
      r.n = Number<LIMBS>::zero();
      return r;
    }

    __host__ __device__ __forceinline__ bool operator==(const Element<LIMBS> &rhs) const &
    {
      return n == rhs.n;
    }

    __host__ __device__ __forceinline__ bool operator!=(const Element<LIMBS> &rhs) const &
    {
      return n != rhs.n;
    }
  };

  template <usize LIMBS>
  struct Params
  {
    u32 m[LIMBS], r_mod[LIMBS], r2_mod[LIMBS];
    u32 m_prime;
  };

  template <usize LIMBS>
  struct Env
  {
    Number<LIMBS> m;
    // m' = -m^(-1) mod b where b = 2^32
    u32 m_prime;
    // r_mod = R mod m,
    // r2_mod = R^2 mod m
    Number<LIMBS> r_mod, r2_mod;
    // m_sub2 = m - 2, for invertion field elements
    Number<LIMBS> m_sub2;

    // Device-Host methods

    __device__ __host__ Env<LIMBS>() {}

    __device__ __host__ __forceinline__
        Element<LIMBS>
        one() const &
    {
      Element<LIMBS> elem;
      elem.n = r_mod;
      return elem;
    }

    // Device methods

    __device__ __forceinline__ Env<LIMBS>(Params<LIMBS> p)
    {
      m = Number<LIMBS>::load(p.m);
      r_mod = Number<LIMBS>::load(p.r_mod);
      r2_mod = Number<LIMBS>::load(p.r2_mod);
      m_prime = p.m_prime;
      auto two = Number<LIMBS>::two();
      m_sub2 = m - two;
    }

    __device__ __forceinline__
        Element<LIMBS>
        mul(const Element<LIMBS> &a, const Element<LIMBS> &b) const &
    {
      Element<LIMBS> r;
      device_arith::montgomery_multiplication<LIMBS>(r.n.limbs, a.n.limbs, b.n.limbs, m.limbs, m_prime);
      return r;
    }

    __device__ __forceinline__
        Element<LIMBS>
        square(const Element<LIMBS> &a) const &
    {
      return mul(a, a);
    }

    __device__ __forceinline__
        Element<LIMBS>
        add(const Element<LIMBS> &a, const Element<LIMBS> &b) const &
    {
      Element<LIMBS> r;
      device_arith::add_modulo<LIMBS>(r.n.limbs, a.n.limbs, b.n.limbs, m.limbs);
      return r;
    }

    __device__ __forceinline__
        Element<LIMBS>
        sub(const Element<LIMBS> &a, const Element<LIMBS> &b) const &
    {
      Element<LIMBS> r;
      device_arith::sub_modulo<LIMBS>(r.n.limbs, a.n.limbs, b.n.limbs, m.limbs);
      return r;
    }

    __device__ __forceinline__
        Element<LIMBS>
        neg(const Element<LIMBS> &a) const &
    {
      if (a.n.is_zero())
        return Element<LIMBS>::zero();
      Element<LIMBS> r;
      device_arith::sub<LIMBS>(r.n.limbs, m.limbs, a.n.limbs);
      return r;
    }

    __device__ __forceinline__
        Element<LIMBS>
        from_number(const Number<LIMBS> &n) const &
    {
      Element<LIMBS> r;
      device_arith::montgomery_multiplication<LIMBS>(r.n.limbs, n.limbs, r2_mod.limbs, m.limbs, m_prime);
      return r;
    }

    __device__ __forceinline__
        Number<LIMBS>
        to_number(const Element<LIMBS> &e) const &
    {
      Number<2 * LIMBS> n;
      memcpy(n.limbs, e.n.limbs, LIMBS * sizeof(u32));
      memset(n.limbs + LIMBS * sizeof(u32), 0, LIMBS * sizeof(u32));
      Number<LIMBS> r;
      device_arith::montgomery_reduction<LIMBS>(n, m.limbs, m_prime);
      memcpy(r.limbs, n.limbs, LIMBS * sizeof(u32));
    }

    __device__ __forceinline__ void
    pow_iter(const Element<LIMBS> &a, bool &found_one, Element<LIMBS> &res, u32 p, u32 deg = 31) const &
    {
#pragma unroll
      for (int i = deg; i >= 0; i--)
      {
        if (found_one)
          res = square(res);
        if ((p >> i) & 1)
        {
          found_one = true;
          res = mul(res, a);
        }
      }
    }

    __device__ __forceinline__
        Element<LIMBS>
        pow(const Element<LIMBS> &a, const Number<LIMBS> &p) const &
    {
      auto res = one();
      bool found_one = false;
#pragma unroll
      for (int i = LIMBS - 1; i >= 0; i--)
        pow_iter(a, found_one, res, p.limbs[i]);
      return res;
    }

    __device__ __forceinline__ Element<LIMBS> pow(const Element<LIMBS> &a, u32 p, u32 deg = 31) const &
    {
      auto res = one();
      bool found_one = false;
      pow_iter(a, found_one, res, p, deg);
      return res;
    }

    __device__ __forceinline__ Element<LIMBS> invert(const Element<LIMBS> &a) const &
    {
      return pow(a, m_sub2);
    }

    // Host methods

    static __host__
        Env<LIMBS>
        host_new(Params<LIMBS> p)
    {
      Env<LIMBS> env;
      env.m = Number<LIMBS>::load(p.m);
      env.r_mod = Number<LIMBS>::load(p.r_mod);
      env.r2_mod = Number<LIMBS>::load(p.r2_mod);
      env.m_prime = p.m_prime;
      auto two = Number<LIMBS>::two();
      env.m_sub2 = env.m.host_sub(two);
      return env;
    }

    __host__ Element<LIMBS> host_sub(const Element<LIMBS> &a, const Element<LIMBS> &b)
    {
      Element<LIMBS> r;
      host_arith::sub_modulo<LIMBS>(r.n.limbs, a.n.limbs, b.n.limbs, m.limbs);
      return r;
    }
    __host__ Element<LIMBS> host_add(const Element<LIMBS> &a, const Element<LIMBS> &b)
    {
      Element<LIMBS> r;
      host_arith::add_modulo<LIMBS>(r.n.limbs, a.n.limbs, b.n.limbs, m.limbs);
      return r;
    }

    __host__ Element<LIMBS> host_neg(const Element<LIMBS> &a) const &
    {
      if (a.n.is_zero())
        return Number<LIMBS>::zero();
      Element<LIMBS> r;
      r.n = m.host_sub(a.n);
      return r;
    }

    __host__
        Element<LIMBS>
        host_mul(const Element<LIMBS> &a, const Element<LIMBS> &b) const &
    {
      Element<LIMBS> r;
      host_arith::montgomery_multiplication<LIMBS>(r.n.limbs, a.n.limbs, b.n.limbs, m.limbs, m_prime);
      return r;
    }

    __host__
        Element<LIMBS>
        host_from_number(const Number<LIMBS> &n) const &
    {
      Element<LIMBS> r;
      host_arith::montgomery_multiplication<LIMBS>(r.n.limbs, n.limbs, r2_mod.limbs, m.limbs, m_prime);
      return r;
    }

    __host__
        Number<LIMBS>
        host_to_number(const Element<LIMBS> &e) const &
    {
      Number<2 * LIMBS> n;
      memcpy(n.limbs, e.n.limbs, LIMBS * sizeof(u32));
      memset(n.limbs + LIMBS * sizeof(u32), 0, LIMBS * sizeof(u32));
      Number<LIMBS> r;
      host_arith::montgomery_reduction(n, m.limbs, m_prime);
      memcpy(r.limbs, n.limbs, LIMBS * sizeof(u32));
    }

    __host__
        Element<LIMBS>
        host_square(const Element<LIMBS> &a) const &
    {
      return host_mul(a, a);
    }

    __host__ void host_pow_iter(const Element<LIMBS> &a, bool &found_one, Element<LIMBS> &res, u32 p, u32 deg = 31) const &
    {
      for (int i = 31; i >= 0; i--)
      {
        if (found_one)
          res = host_square(res);
        if ((p >> i) & 1)
        {
          found_one = true;
          res = host_mul(res, a);
        }
      }
    }

    __host__ Element<LIMBS> host_pow(const Element<LIMBS> &a, const Number<LIMBS> &power) const &
    {
      auto res = one();
      bool found_one = false;
#pragma unroll
      for (int i = LIMBS - 1; i >= 0; i--)
        host_pow_iter(a, found_one, res, power.limbs[i]);
      return res;
    }

    __host__ Element<LIMBS> host_pow(const Element<LIMBS> &a, const u32 &power)
    {
      auto res = one();
      bool found_one = false;
      host_pow_iter(found_one, res, power);
      return res;
    }
  };

  template <usize LIMBS>
  std::istream &
  operator>>(std::istream &is, Number<LIMBS> &n)
  {
    char _;
    is >> _ >> _;
    for (int i = LIMBS - 1; i >= 0; i--)
      is >> std::hex >> n.limbs[i] >> _;
    return is;
  }

  template <usize LIMBS>
  std::ostream &
  operator<<(std::ostream &os, const Number<LIMBS> &n)
  {
    os << "0x";
    for (usize i = LIMBS - 1; i >= 1; i--)
      os << std::hex << std::setfill('0') << std::setw(8) << n.limbs[i] << '_';
    os << std::hex << std::setfill('0') << std::setw(8) << n.limbs[0];
    return os;
  }

}

#endif