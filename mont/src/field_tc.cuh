#ifndef FIELD_TC_H
#define FIELD_TC_H

#include "./field.cuh"
#include "./ptx.cuh"

namespace mont
{
  namespace tc256
  {

    using Stage = u8;

    // Load the consecutive four bytes of storage `p` starting at the `n`-th byte.
    // `p` has `n_limbs` u32.
    template <typename StL>
    __device__ __forceinline__ u32 unanligned_u32_access(StL p, u32 n_limbs, int n)
    {
      auto get_limb = [p, n_limbs](int i)
      {
        if (i < 0 || i >= n_limbs)
          return 0U;
        else
          return p[i];
      };
      int rem = n % 4;
      int n0 = n - rem;
      if (rem == 0)
        return get_limb(n / 4);
      else
        return (get_limb(n0 / 4) >> (rem * 8)) | (get_limb(n0 / 4 + 1) << (32 - rem * 8));
    }

    // Let `x` be composed of four bytes in little-endian `[x0, x1, x2, x3]`,
    // returns `[x3, x2, x1, x0]`
    __device__ __forceinline__ u32 bytes_reversed(u32 x)
    {
      return __byte_perm(x, 0, 0x00010203);
    }

    // Layout of the 16x32 A matrix in mma.m16n8k32.s32 (or called A layout) is
    //
    //         0   1   2   3                4 ... 7    8 ... 11    12 ... 15     16    17    18   19            20 ... 23    ...
    //   0  T0{a0[0],a0[1],a0[2],a0[3]}     T1         T2          T3         T0{a1[0],a1[1],a1[2],a1[3]}       T1
    //   1  T4                              T5         T6          T7         T4                                T5
    //      ...
    //   7  T28                             T29        T30         T31        T28                               T29
    //   8  T0{a2[0],a2[1],a2[2],a2[3]}     T1         T2          T3         T0{a3[0],a3[1],a3[2],a3[3]}       T1
    //   .
    //   15 T28                             T29        T30         T31        T20                               T29
    //
    // where a0[0] indicates the lowest byte of u32 a0.
    // In this illustration, T0 means the 0-th lane, and a0, a1, a2, a3 are four u32's in 0-th lane.
    // For therad i, Arguments `a0`, `a1`, `a2`, `a3` correspond to "a0, a1, a2, a3" in the illustration, at `i % 32`-th lane.
    //
    // This function loads big number in `limb` to A matrix, in a way prepared for convolution.
    // For example, following is how a big number of 4 bytes is loaded into a 8x4 matrix for convolution.
    //
    //   x0
    //   x1  x0
    //   x2  x1  x0
    //   x3  x2  x1  x0
    //       x3  x2  x1
    //           x3  x2
    //               x3
    //               0
    //
    // In the actual case of big number of 32 bytes, the matrix is splitted into four 16x32 matrices vertically,
    // and `stage` determines which one is loaded into A.
    template <typename StL>
    __device__ __forceinline__ void load_a_matrix(
        u32 &a0, u32 &a1, u32 &a2, u32 &a3,
        StL limbs, u32 n_limbs, Stage stage)
    {
      u32 lane_id = threadIdx.x % 32;

      auto get = [lane_id, stage, limbs, n_limbs](u32 m, u32 n)
      {
        u32 i = (lane_id / 4) + stage * 16 + m * 8;
        u32 j = (lane_id % 4) * 4 + 3 + n * 16;
        return bytes_reversed(unanligned_u32_access(limbs, n_limbs, i - j));
      };

      a0 = get(0, 0);
      a1 = get(0, 1);
      a2 = get(1, 0);
      a3 = get(1, 1);
    }

    // Layout of 32x8 B matrix in mma.m16n8k32.s32 (or called B layout) is
    //
    //               0           1    2   ...   7
    //  0           {b0[0],
    //  1         T0 b0[1],      T4   T8  ...   T28
    //  2            b0[2],
    //  3            b0[3]}
    //  4 ... 7   T1             T5   T9  ...   T29
    //  8 ...11   T2             T6   T10 ...   T30
    //  12...15   T3             T7   T11 ...   T31
    //  16          {b1[0],
    //  17        T0 b1[1],      T4   T8  ...   T28
    //  18           b1[2],
    //  19           b1[3]}
    //  20...23   T1             T5   T9  ...   T29
    //            ...
    //
    // In this illustration, T0 means the 0-th lane, and b0, b1 are two u32's in 0-th lane.
    //
    // This function loads at most four big number in `p` to B matrix, in four interleaved columns.
    // Four numbers of 32 bytes x1, x2, x3, x4 are loaded into the 32x8 matrix B
    //
    //  x1[0]  0  x1[0]  0  x2[0]  0  x3[0]  0
    //  x1[1]  0  x1[1]  0  x2[1]  0  x3[1]  0
    //  ...
    //  x1[31] 0  x1[31] 0  x2[31] 0  x3[31] 0
    template <u32 NUM, typename StL>
    __device__ __forceinline__ void load_b_matrix(
        u32 &b0, u32 &b1,
        const StL *p)
    {
      u32 lane_id = threadIdx.x % 32;
      u32 col = lane_id / 4;
      u32 j = col / 2;
      if (col % 2 == 0 && j + 1 >= NUM)
      {
        u32 i = lane_id % 4;
        b0 = p[j][i];
        b1 = p[j][i + 4];
      }
    }

    // Computes D = A * B + C where A is 16x32, B is 32x8, C and D are 32x8.
    // Layout of C and D (or called D layout) is
    //
    //          0  1    2..3  4..5  6..7
    //   0   T0{d0,d1}  T1    T2    T3
    //   1   T4         T5    T6    T7
    //       ...
    //   7   T28        T29   T30   T31
    //   8   T0{d2,d3}  T1    T2    T3
    //       ...
    //   15  T28        T29   T30   T31
    //
    // where T0 means the 0-th lane, and d0, d1, d2, d3 are four i32's in 0-th lane.
    __device__ __forceinline__ void mma_m16n8k32(
        u32 &d0, u32 &d1, u32 &d2, u32 &d3,
        u32 a0, u32 a1, u32 a2, u32 a3,
        u32 b0, u32 b1,
        u32 c0, u32 c1, u32 c2, u32 c3)
    {
      i32 d0i, d1i, d2i, d3i;
      i32 c0i = c0, c1i = c1, c2i = c2, c3i = c3;
      asm(
          "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};" :
          "=r"(d0i), "=r"(d1i), "=r"(d2i), "=r"(d3i) : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(c0i), "r"(c1i), "r"(c2i), "r"(c3i));
      d0 = d0i, d1 = d1i, d2 = d2i, d3 = d3i;
    }

    // Transpose a 8x8 matrix, of data type u16.
    //
    //          0    1      2..3  4..5  6..7
    //   0   T0{x[0],x[1]}  T1    T2    T3
    //   1   T4{y[0],y[1]}  T5    T6    T7
    //       ...
    //   7   T28            T29   T30   T31
    //
    // The matrix above is tranposed to
    //
    //          0  1        2..3  4..5  6..7
    //   0   T0{x[0],y[0]}  T1    T2    T3
    //   1   T4{x[1],y[1]}  T5    T6    T7
    //       ...
    //   7   T28            T29   T30   T31
    //
    // Then the two u16's in some lane is returned as u32, in littler-endian. For example, lane 0 returns x[0] | (y[0] << 16)
    __device__ __forceinline__ u32 transpose_m8n8(u32 a)
    {
      u32 r;
      asm(
          "movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(r) : "r"(a));
      return r;
    }

    // To move some big integer in D layout to B layout for later multiplication, the following transposition is carried out.
    // Let 32x8 matrix D[ij] (column major) =
    //
    //          0   1     2..3  4..5  6..7
    //   0   T0{d00,d00}  T1    T2    T3
    //   1   T4           T5    T6    T7
    //       ...
    //   7   T28          T29   T30   T31
    //   8   T0{d02,d02}  T1    T2    T3
    //       ...
    //   15  T28          T29   T30   T31
    //   16  T0{d10,d10}  T1    T2    T3
    //       ...
    //   23  T28          T29   T30   T31
    //   24  T0{d12,d12}  T1    T2    T3
    //       ...
    //   31  T28          T29   T30   T31
    //
    // It represents four big integers, in column 0, 2, 4, 6, respectively.
    // D[ij] is the j-th byte of the i-th big integer.
    // D[ij] is transposed to Bx0 =
    //
    //       0                               1                        2                         3                ...        7
    //   0   T0(b00=D[0,0],b01=D[0,1])       T4(D[0,0],D[0,1])        T8 (D[2,0],D[2,1])        T12(D[2,0],D[2,1])          T28
    //   1   T1(D[0,2],D[0,3])               T5(D[0,2],D[0,3])        T9 (D[2,2],D[2,3])        T13(D[2,2],D[2,3])          T29
    //   2   T2(D[0,4],D[0,5])               T6(D[0,4],D[0,5])        T10(D[2,4],D[2,5])        T14(D[2,4],D[2,5])          T30
    //   3   T3(D[0,6],D[0,7])               T7(D[0,6],D[0,7])        T11(D[2,6],D[2,7])        T15(D[2,6],D[2,7])          T31
    //   4   T0(b10=D[0,9],b11=D[0,10])
    //       ...
    //   7   T3(D[0,14],D[0,15])
    //   8   T0(b20=D[0,16],b21=D[0,17])
    //       ...
    //   11  T3(D[0,22],D[0,23])
    //   12  T0(b30=D[0,24],b31=D[0,25])
    //       ...
    //   15  T3(D[0,30],D[0,31])
    //
    // If mma is called with C = 0, then each D[ij] is at most 2^21, so the highest byte of D[ij] is zero.
    // This way, for example, D[0,0] and D[0,1] can be coalesced into a u32
    //   Bx[0,0] = D[0,0] + (D[0,1] << 8)
    // Note that D' is not calculated in this function.
    __device__ __forceinline__ void transpose_d_to_b(
        u32 &b00, u32 &b01, u32 &b10, u32 &b11, u32 &b20, u32 &b21, u32 &b30, u32 &b31,
        u32 d00, u32 d02,
        u32 d10, u32 d12)
    {
      auto calc = [](u32 &b0, u32 &b1, u32 d)
      {
        u16 d_lo, d_hi;
        ptx::unpack_b32(d_lo, d_hi, d);
        u32 b_lo = transpose_m8n8(ptx::pack_b32(d_lo, d_lo));
        u16 b0_lo, b1_lo, b0_hi, b1_hi;
        ptx::unpack_b32(b0_lo, b1_lo, b_lo);
        u32 b_hi = transpose_m8n8(ptx::pack_b32(d_hi, d_hi));
        ptx::unpack_b32(b0_hi, b1_hi, b_hi);
        b0 = ptx::pack_b32(b0_lo, b0_hi);
        b1 = ptx::pack_b32(b1_lo, b1_hi);
      };

      calc(b00, b01, d00);
      calc(b10, b11, d02);
      calc(b20, b21, d10);
      calc(b30, b31, d12);
    }

    // After `transpose_d_to_b`, the number to be multiplied has been transposed to layout similar to B layout.
    // However, B matrix in mma requries data type u8, so the matrix Bx mentioned in `transpose_d_to_be` needs further shuffling.
    // In this scheme, columns 0,2,4,6 are each splitted into two columns of u8's, one column in 0,2,4,6 and the other in 1,3,5,7
    // Consider first column Bx[0], it represents a big integer
    //   N(Bx[0]) = Bx[0,0] + Bx[0,1] * 2^16 + Bx[0,2] * 2^32 + ... + Bx[0,15] * 2^240
    // If we take the terms at even position
    //   N_0(Bx[0]) = Bx[0,0] + Bx[0,2] * 2^32 + ... + Bx[0,14] * 2^224
    // Recall that Bx[ij] are all of u32's, they can constitute a column in B layout that represents a big integer.
    // Similarly, let
    //   Bx'[0,1] = Bx[0,1]_hi
    //   Bx'[0,2 * i - 1] = Bx[0,2 * i - 3]_hi + (Bx[0,2 * i - 1]_lo << 16), i >= 2
    // where _hi and _lo means the higher or lower u16 of the u32, then
    //   N_1(Bx[0]) = Bx'[0,1] + Bx'[0,3] * 2^32 + ... + Bx'[0,15] * 2^224
    // also constitute a column in B layout that represents a big integer.
    // Note that N_0(Bx[0]) + N_1(Bx[0]) does not equal the original N(Bx[0]), since the higher u16 of Bx[0,15] is thrown away,
    // but since montgomery reduction only needs modular 2^256 multiplication when calculating u = m' * (a * b), this is not a problem.
    //
    // The resulting By matrix is
    //
    //              0                           1                   2       3    ...   7
    //   0         {by0[0]=Bx[0,0][0],         {Bx'[0,1][0],
    //   1       T0 by0[1]=Bx[0,0][1],       T4 Bx'[0,1][1],        T8      T9         T28
    //   2          by0[2]=Bx[0,0][2],          Bx'[0,1][2],
    //   3          by0[3]=Bx[0,0][3]}          Bx'[0,1][3]}
    //   4 .. 7  T1{Bx[0,2]}                 T5{Bx'[0,3]}
    //   8 ..11  T2{Bx[0,4]}                 T6{Bx'[0,5]}
    //   12..15  T3{Bx[0,6]}                 T7{Bx'[0,7]}
    //   16        {by1[0]=Bx[0,8][0],         {Bx'[0,9][0],
    //   17      T0 by1[1]=Bx[0,8][1],       T4 Bx'[0,9][1],
    //   18         by1[2]=Bx[0,8][2],          Bx'[0,9][2],
    //   19         by1[3]=Bx[0,8][3]}          Bx'[0,9][3]}
    //   20..23  T1{Bx[0,10]}                T5{Bx'[0,11]}
    //   24..27  T2{Bx[0,12]}                T6{Bx'[0,13]}
    //   28..31  T3{Bx[0,14]}                T7{Bx'[0,15]}
    //
    __device__ __forceinline__ void shuffle_b_for_mul(
        u32 &by0, u32 &by1,
        u32 bx0, u32 bx1, u32 bx2, u32 bx3)
    {
      u32 lane_id = threadIdx.x % 32;
      u32 i = lane_id % 4;
      u32 col = lane_id / 4;
      u32 j = col / 2;
      u32 k = col % 2;
      u32 mask = 0xffffffff;

      u16 bx0_hi, bx0_lo;
      u16 bx1_hi, bx1_lo;
      u16 bx2_hi, bx2_lo;
      u16 bx3_hi, bx3_lo;
      ptx::unpack_b32(bx0_lo, bx0_hi, bx0);
      ptx::unpack_b32(bx1_lo, bx1_hi, bx1);
      ptx::unpack_b32(bx2_lo, bx2_hi, bx2);
      ptx::unpack_b32(bx3_lo, bx3_hi, bx3);

      u32 src_var;

      // Round 1
      if (i % 2 == 0)
        src_var = bx0;
      else
        src_var = ptx::pack_b32(bx0_hi, bx1_hi);

      u32 src_lane = (col - k) * 4 + (i * 2 + k) % 4;
      // if (k == 0)
      //   src_lane = col * 4 + i * 2;
      // else
      //   src_lane = (col - 1) * 4 + (i * 2 + 1) % 4;

      u32 got = __shfl_sync(mask, src_var, src_lane);
      u16 got_hi, got_lo;
      ptx::unpack_b32(got_lo, got_hi, got);
      if (k == 0 && i < 2)
        by0 = got;
      else if (k == 1 && i < 2)
        by0 |= got_lo << 16;
      else if (k == 1 && i >= 2)
        by0 |= got_hi << 16;

      // Round 2
      if (i % 2 == 0)
        src_var = bx1;
      else
        src_var = ptx::pack_b32(bx0_lo, bx1_lo);

      src_lane = (col - k) * 4 + (i * 2 - k) % 4;

      got = __shfl_sync(mask, src_var, src_lane);
      ptx::unpack_b32(got_lo, got_hi, got);

      if (k == 0 && i >= 2)
        by0 = got;
      else if (k == 1 && i == 0)
        by1 |= got_hi;
      else if (k == 1 && i == 1)
        by0 |= got_lo;
      else if (k == 1 && i == 2)
        by0 |= got_lo;
      else if (k == 1 && i == 3)
        by0 |= got_hi;

      // Round 3
      if (i % 2 == 0)
        src_var = bx2;
      else
        src_var = ptx::pack_b32(bx2_hi, bx3_hi);

      src_lane = (col - k) * 4 + (i * 2 + k) % 4;

      got = __shfl_sync(mask, src_var, src_lane);
      ptx::unpack_b32(got_lo, got_hi, got);

      if (k == 0 && i < 2)
        by1 = got;
      else if (k == 1 && i < 2)
        by1 |= got_lo << 16;
      else if (k == 1 && i >= 2)
        by1 |= got_hi << 16;

      // Round 4
      if (i % 2 == 0)
        src_var = bx3;
      else
        src_var = ptx::pack_b32(bx2_lo, bx3_lo);

      src_lane = (col - k) * 4 + (i * 2 - k) % 4;

      got = __shfl_sync(mask, src_var, src_lane);
      ptx::unpack_b32(got_lo, got_hi, got);

      if (k == 0 && i >= 2)
        by1 = got;
      else if (k == 1 && i == 1)
        by1 |= got_lo;
      else if (k == 1 && i == 2)
        by1 |= got_lo;
      else if (k == 1 && i == 3)
        by1 |= got_hi;
    }

    // This function composes `transpose_d_to_b` and `shuffle_b_for_mul`.
    // It first transposes D to Bx0, then Bx0's columns are coalesced as described before,
    // resulting in Bx;
    // After that By is obtained by applying `shuffle_b_for_mul` to Bx.
    __device__ __forceinline__ void transpose_and_split(
        u32 &by0, u32 &by1,
        u32 d00, u32 d02,
        u32 d10, u32 d12)
    {
      u32 db00, db01, db10, db11, db20, db21, db30, db31;
      transpose_d_to_b(
          db00, db01, db10, db11, db20, db21, db30, db31,
          d00, d02, d10, d12);
      shuffle_b_for_mul(
          by0, by1,
          db00 + (db01 << 8),
          db10 + (db11 << 8),
          db20 + (db21 << 8),
          db30 + (db31 << 8));
    }

    // Montgomery reduction zeros-out lower 256 bit of d = a * b, but the matrix column representing d consists of i32 elements,
    //   d = d[0] + d[1] * 2^8 + ... + d[30] * 2^240 + d[31] * 2^248 + d[32] * 2^256 + ... + d[63] * 2^504
    // so d[30]'s third byte (also the highest non-zero byte) has weight 2^256, and d[31]'s second and third byte has weight 2^256, 2^264.
    // They are non-zero after applying montgomery reduction to them.
    // Therefore these bytes must be added to d[32], which is what this function does.
    // After mma, lane 28's d12 holds d[30] and lane24's d12 holds d[31].
    __device__ __forceinline__ void carry_up_tail_of_d(
        u32 &d20, u32 d12)
    {
      u32 lane_id = threadIdx.x % 32;
      u32 i = lane_id / 4;
      u32 mask = 0xf000000f;
      u32 got = __shfl_sync(mask, d12, lane_id % 4 + 28);
      if (i == 0)
        d20 += got >> 16;

      mask = 0x0f00000f;
      got = __shfl_sync(mask, d12, lane_id % 4 + 28);
      if (i == 0)
        d20 += got >> 8;
    }

    __device__ __forceinline__ void shuffle_e_for_compact(
        u32 &ey0, u32 &ey1,
        u32 ex0, u32 ex1, u32 ex2, u32 ex3)
    {
      u32 lane_id = threadIdx.x % 32;
      u32 i = lane_id / 4;
      u32 j = lane_id % 4;
      u32 mask = 0xffffffff;

      auto shuffle = [i, j, mask](u32 &ey, u32 exx0, u32 exx1)
      {
        // Round 1
        u32 src_lane = i * 8 + j;
        u32 got = __shfl_sync(mask, exx0, src_lane);
        if (i < 4)
          ey = got;

        // Round 2
        src_lane = (i - 4) * 8 + j;
        got = __shfl_sync(mask, exx1, src_lane);
        if (i >= 4)
          ey = got;
      };

      shuffle(ey0, ex0, ex1);
      shuffle(ey1, ex2, ex3);
    }

    template <bool IS_SUB = false>
    __device__ __forceinline__ void fast_propagate_u16(
        u16 &r, u32 carry_out, u32 &carry_in)
    {
      u32 lane_id = threadIdx.x % 32;
      u32 i = lane_id / 4;
      u32 j = lane_id % 4;
      u32 mask = 0xffffffff;

      u32 full = IS_SUB ? r == 0 : r == 0xffff;

      u32 lco_b = __ballot_sync(mask, carry_out);
      u32 fu_b = __ballot_sync(mask, full);
      lco_b = lco_b & (0x11111111 << j);
      fu_b = fu_b | (0xeeeeeeee << j);
      lco_b = (lco_b << 4) | (carry_in << j);

      u32 carries = ptx::add_cc(lco_b, fu_b);
      carry_in = ptx::addc(carry_in, 0);

      if ((carries ^ fu_b) & (1 << lane_id))
      {
        if (IS_SUB)
          r -= 1;
        else
          r += 1;
      }
    }

    // For the result t of montgomery reduction
    //   t = t[0] + t[1] * 2^8 + ... + t[31] * 2^248
    // where d[i] is of i32, it must be transformed into
    //   r = r[0] + r[1] * 2^16 + ... + r[15] * 2^240
    // where r[i] is of u16. This function implements this transformation.
    //
    // Input matrix has D layout
    //
    //          0           1        2..3  4..5  6..7
    //   0   T0 {d00=t[0]  ,d01}      T1    T2    T3
    //   1   T4 {t[1]      ,_  }      T5    T6    T7
    //       ...
    //   7   T28{t[7]      ,_  }      T29   T30   T31
    //   8   T0 {d02=t[8]  ,d03}      T1    T2    T3
    //       ...
    //   15  T28{t[15]     ,_  }      T29   T30   T31
    //   16  T0 {d10=t[16] ,d11}      T1    T2    T3
    //   17  T4 {t[17]     ,_  }      T5    T6    T7
    //       ...
    //   23  T28{t[12]     ,_  }      T29   T30   T31
    //   24  T0 {d12=t[24] ,d13}      T1    T2    T3
    //       ...
    //   31  T28{t[31]     ,_  }      T29   T30   T31
    //
    // And the output matirx has Z layout
    //
    //           0           1    2    4
    //   0   T0 {r0=r[0] }   T1   T2   T3
    //   1   T4 {r[1]    }   T5   T6   T7
    //       ...
    //   3   T12{r[3]    }   T13  T14  T15
    //   4   T16{r[4]    }   T17  T18  T19
    //       ...
    //   7   T28{r[7]    }   T29  T30  T31
    //   8   T0 {r1=r[8] }
    //       ...
    //   11  T12{r[11]   }
    //   12  T16{r[12]   }
    //       ...
    //   15  T28{r[15]   }
    //
    __device__ __forceinline__ void compact(
        u16 &r0, u16 &r1,
        u32 d00, u32 d02,
        u32 d10, u32 d12)
    {
      u32 lane_id = threadIdx.x % 32;
      u32 i = lane_id / 4;
      u32 j = lane_id % 4;
      u32 mask = 0xffffffff;
      u32 d00_higher = __shfl_down_sync(mask, d00, 4);

      u32 e0, e1, e2, e3;

      if (i % 2 == 0)
        e0 = d00 + (d00_higher << 8);

      u32 d02_higher = __shfl_down_sync(mask, d02, 4);
      if (i % 2 == 0)
        e1 = d02 + (d02_higher << 8);

      u32 d10_higher = __shfl_down_sync(mask, d10, 4);
      if (i % 2 == 0)
        e2 = d10 + (d10_higher << 8);

      u32 d12_higher = __shfl_down_sync(mask, d12, 4);
      if (i % 2 == 0)
        e3 = d12 + (d12_higher << 8);

      u32 f0, f1;
      shuffle_e_for_compact(f0, f1, e0, e1, e2, e3);

      if (i == 7)
      {
        f1 = (f1 & 0x0000ffff) | (f0 & 0xffff0000);
        f0 = f0 & 0x0000ffff;
      }

      u32 carry_in = 0;
      auto calc = [&carry_in, mask, lane_id, i, j](u16 &r, u32 &f)
      {
        u32 e_lower = __shfl_up_sync(mask, f, 4);
        u32 r_wide = (e_lower >> 16) + (f & 0x0000ffff);
        r = r_wide & 0x0000ffff;
        u32 limb_carry_out = r_wide >> 16;

        fast_propagate_u16(r, limb_carry_out, carry_in);
      };

      calc(r0, f0);
      calc(r1, f1);
    }

    // Compute z = r mod m.
    // z and r are both in Z layout.
    template <typename StM>
    __device__ __forceinline__ void modulo_m(
        u16 &z0, u16 &z1,
        u16 r0, u16 r1,
        StM st_m)
    {
      u32 lane_id = threadIdx.x % 32;
      u32 i = lane_id / 4;
      u32 j = lane_id % 4;
      u32 mask = 0xffffffff;
      u32 borrow_in = 0;

      auto get_st_u16 = [st_m](u32 n)
      {
        u32 limb = st_m[n / 2];
        return n % 2 == 0 ? limb & 0x0000ffff : limb >> 16;
      };

      auto sub = [&borrow_in, mask, lane_id, i, j, get_st_u16](u16 &z, u16 r, u32 round)
      {
        u32 z_wide = r - get_st_u16(i + round * 8);
        z = z_wide & 0x0000ffff;
        u32 limb_borrow = z_wide >> 31;

        fast_propagate_u16<true>(z, limb_borrow, borrow_in);
      };

      u32 carry_in = 0;
      auto add = [&carry_in, mask, lane_id, i, j, get_st_u16](u16 &z, u32 round)
      {
        u32 z_wide = z + get_st_u16(i + round * 8);
        z = z_wide & 0x0000ffff;
        u32 limb_carry_out = z_wide >> 16;

        fast_propagate_u16(z, limb_carry_out, carry_in);
      };

      sub(z0, r0, 0);
      sub(z1, r1, 1);

      if (borrow_in)
      {
        add(z0, 0);
        add(z0, 1);
      }
    }

    // Store z to st_z, z is in Z layout.
    template <u32 NUM, typename StZ>
    __device__ __forceinline__ void store_z(
        StZ *st_z,
        u16 z0, u16 z1)
    {
      u32 lane_id = threadIdx.x % 32;
      u32 i = lane_id / 4;
      u32 j = lane_id % 4;
      u32 mask = 0xffffffff;

      u32 w = 0;

      u32 send = ((u32)z1 << 16) + z0;
      u32 got = __shfl_sync(mask, send, (i % 4) * 8 + j);
      if (i < 4)
        w |= got & 0x0000ffff;
      else
        w |= got >> 16;

      got = __shfl_sync(mask, send, (i % 4) * 8 + 4 + j);
      if (i < 4)
        w |= got & 0xffff0000;
      else
        w |= got << 16;

      if (j < NUM)
        st_z[j][i] = w;
    }

    // Debugging utilies
    namespace debug {
      template <u32 R, u32 C, typename T>
      struct Matrix {
        T* p;
        __host__ Matrix() {
          cudaMalloc(&p, R * C * sizeof(T));
        }
        __device__ __host__ T& operator[](u32 i, u32 j)
        {
          return p[i * C + j];
        }
      };

      // Put a A layotu matrix into memory
      __device__ __forceinline__ void store_a_matrix(
        u32 a0, u32 a1, u32 a2, u32 a3,
        Matrix<16, 32, u8> mat
      ) {
        u32 lane_id = threadIdx.x % 32;

        auto st = [mat, lane_id](u32 m, u32 n, u32 a)
        {
            u32 i = lane_id / 4 + m * 8;
            u32 j = (lane_id % 4) * 4 + n * 16;
            for (u32 k = 0; k < 4; k ++)
              m[i, j + k] = (a >> (8 * k)) & 0xff;
        };
        st(0, 0, a0);
        st(0, 1, a1);
        st(1, 0, a2);
        st(1, 1, a3);
      }

      // Put a B layotu matrix into memory
      __device__ __forceinline__ void store_b_matrix(
        u32 b0, u32 b1, Matrix<31, 8, u8> mat
      ) {
        u32 lane_id = threadIdx.x % 32;

        auto st = [mat, lane_id](u32 m, u32 b)
        {
          u32 i = (lane_id % 4) * 4 + m * 8;
          u32 j = lane_id / 4;
          for (u32 k = 0; k < 4; k ++)
            m[i + k, j] = (b >> (8 * k)) & 0xff;
        };

        st(0, b0);
        st(1, b1);
      }
    }

    template <u32 NUM, typename StZ, typename StX, typename StY, typename StM>
    __device__ __forceinline__ void montgomery_multiplication_raw(
        StZ *st_z,
        const StX st_x,
        const StY *st_y,
        const StM st_m,
        const StM st_m_prime)
    {
      // Naming convention: <symbol> <layout> [variant]   <number>
      // For example,        s        d                    00
      //                     mp       a                    0
      //                     u        b        s(plitted)  0

      // Compute s = x * y
      u32 xa0, xa1, xa2, xa3;
      load_a_matrix(xa0, xa1, xa2, xa3, st_x, 8, 0);
      u32 yb0, yb1;
      load_b_matrix<NUM>(yb0, yb1, st_y);
      u32 sd00, sd01, sd02, sd03;
      mma_m16n8k32(sd00, sd01, sd02, sd03, xa0, xa1, xa2, xa3, yb0, yb1, 0, 0, 0, 0);

      load_a_matrix(xa0, xa1, xa2, xa3, st_x, 8, 1);
      u32 sd10, sd11, sd12, sd13;
      mma_m16n8k32(sd10, sd11, sd12, sd13, xa0, xa1, xa2, xa3, yb0, yb1, 0, 0, 0, 0);

      load_a_matrix(xa0, xa1, xa2, xa3, st_x, 8, 2);
      u32 sd20, sd21, sd22, sd23;
      mma_m16n8k32(sd20, sd21, sd22, sd23, xa0, xa1, xa2, xa3, yb0, yb1, 0, 0, 0, 0);

      load_a_matrix(xa0, xa1, xa2, xa3, st_x, 8, 3);
      u32 sd30, sd31, sd32, sd33;
      mma_m16n8k32(sd30, sd31, sd32, sd33, xa0, xa1, xa2, xa3, yb0, yb1, 0, 0, 0, 0);

      // Compute u = m' * s (mod 2^256)

      u32 sbs0, sbs1;
      transpose_and_split(sbs0, sbs1, sd00, sd02, sd10, sd12);

      u32 mpa0, mpa1, mpa2, mpa3;
      load_a_matrix(mpa0, mpa1, mpa2, mpa3, st_m_prime, 8, 0);

      u32 ud00, ud01, ud02, ud03;
      mma_m16n8k32(ud00, ud01, ud02, ud03, mpa0, mpa1, mpa2, mpa3, sbs0, sbs1, 0, 0, 0, 0);

      u32 ud10, ud11, ud12, ud13;
      mma_m16n8k32(ud10, ud11, ud12, ud13, mpa0, mpa1, mpa2, mpa3, sbs0, sbs1, 0, 0, 0, 0);

      ud00 += ud01;
      ud02 += ud03;
      ud10 += ud11;
      ud12 += ud13;

      // Compute t = s + m * u

      u32 ubs0, ubs1;
      transpose_and_split(ubs0, ubs1, ud00, ud02, ud10, ud12);

      u32 ma0, ma1, ma2, ma3;
      load_a_matrix(ma0, ma1, ma2, ma3, st_m, 8, 1);
      u32 td10, td11, td12, td13;
      mma_m16n8k32(td10, td11, td12, td13, ma0, ma1, ma2, ma3, ubs0, ubs1, sd10, 0, sd12, 0);

      u32 td20, td21, td22, td23;
      mma_m16n8k32(td20, td21, td22, td23, ma0, ma1, ma2, ma3, ubs0, ubs1, sd20, 0, sd22, 0);

      u32 td30, td31, td32, td33;
      mma_m16n8k32(td30, td31, td32, td33, ma0, ma1, ma2, ma3, ubs0, ubs1, sd30, 0, sd32, 0);

      td10 += td11;
      td12 += td13;
      td20 += td21;
      td22 += td23;
      td30 += td31;
      td32 += td33;

      // Compute z = t mod 2^256

      carry_up_tail_of_d(td20, td12);

      u16 r0, r1;
      compact(r0, r1, td20, td22, td30, td32);

      u16 z0, z1;
      modulo_m(z0, z1, r0, r1, st_m);

      store_z<NUM>(st_z, z0, z1);
    }

    template <u32 NUM, typename Params, typename StZ, typename StX, typename StY>
    __device__ __forceinline__ void mul(
        Element<Params, StZ> *z,
        Element<Params, StX> x,
        Element<Params, StY> *y)
    {
      StZ st_z[4] = {z[0].n.limbs, z[1].n.limbs, z[2].n.limbs, z[3].n.limbs};
      StY st_y[4] = {y[0].n.limbs, y[1].n.limbs, y[2].n.limbs, y[3].n.limbs};
      montgomery_multiplication_raw<NUM>(
          st_z,
          x.n.limbs,
          st_y,
          Params::m().limbs,
          Params::m_prime_wide().limbs);
    }
  }
}

#endif