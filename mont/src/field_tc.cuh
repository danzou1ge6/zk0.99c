#ifndef FIELD_TC_H
#define FIELD_TC_H

#include <iostream>

#include "./field.cuh"
#include "./ptx.cuh"

namespace mont
{
  namespace tc256
  {

    using Stage = u8;
    const u32 MASK_ALL = 0xffffffff;

    // Let `x` be composed of four bytes in little-endian `[x0, x1, x2, x3]`,
    // returns `[x3, x2, x1, x0]`
    __device__ __forceinline__ u32 bytes_reversed(u32 x)
    {
      return __byte_perm(x, 0, 0x00000123);
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
    //
    // CAUTION that here `limbs` is assumed to be a 24 u32's array, with the 8 u32's containing the big number lies
    // in u32's 8, 9, ..., 15
    __device__ __forceinline__ void load_a_matrix_shared(
        u32 &a0, u32 &a1, u32 &a2, u32 &a3,
        const u32 *limbs, u32 n_limbs, Stage stage)
    {
      u32 lane_id = threadIdx.x % 32;
      u32 i = lane_id / 4 + stage * 16;
      u32 j = lane_id % 4;
      u32 rem = (i & 0b11);
      const u32 *p = limbs;
      
      u32 k0 = i / 4 + 8 - j;
      u32 k1 = k0 - 4;
      u32 k2 = k0 + 2;
      u32 k3 = k0 - 2;

      auto f = [rem](u32 &a, u32 lo, u32 hi)
      {
        asm("prmt.b32.b4e %0, %1, %2, %3;": "=r"(a): "r"(hi), "r"(lo), "r"(rem));
      };
      f(a0, p[k0 - 1], p[k0]);
      f(a1, p[k1 - 1], p[k1]);
      f(a2, p[k2 - 1], p[k2]);
      f(a3, p[k3 - 1], p[k3]);
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
    //
    // `(MASK >> j) & 1` decides whether j-th column is to be loaded from p[j]
    template <u32 MASK, typename F>
    __device__ __forceinline__ void load_b_matrix(
        u32 &b0, u32 &b1,
        F p)
    {
      u32 lane_id = threadIdx.x % 32;
      u32 j = lane_id / 8;
      u32 i = lane_id % 4;
      if ((MASK >> j) & 1)
      {
        b0 = p(j, i);
        b1 = p(j, i + 4);
      }
    }

    // Debugging utilies
    namespace debug
    {
      template <u32 R, u32 C, typename T>
      struct Matrix
      {
        T *p;
        Matrix() = default;
        __host__ Matrix(T *p) : p(p) {}
        __host__ void alloc_device()
        {
          cudaMalloc(&p, sizeof(T) * R * C);
        }
        static __host__ Matrix new_host()
        {
          T *p = new T[R * C];
          return Matrix(p);
        }
        __host__ Matrix to_host()
        {
          auto h = new_host();
          cudaMemcpy(h.p, p, R * C * sizeof(T), cudaMemcpyDeviceToHost);
          return h;
        }
        __device__ __host__ T &get(u32 i, u32 j)
        {
          return p[i * C + j];
        }
        __device__ __host__ const T &get(u32 i, u32 j) const
        {
          return p[i * C + j];
        }
      };

      template <u32 R, u32 C, typename T>
      std::ostream &operator<<(std::ostream &os, const Matrix<R, C, T> mat)
      {
        u32 w = std::is_same<T, u8>() ? 3 : 10;
        for (u32 i = 0; i < R; i++)
        {
          for (u32 j = 0; j < C; j++)
          {
            os << std::hex << std::setw(w) << std::setfill(' ') << (u32)mat.get(i, j) << " ";
          }
          os << std::endl;
        }
        return os;
      }

      struct Intermediates
      {
        // Naming convention here is similar to that in `montgomery_multiplication_raw`,
        // but here the number distinguishes different matrix chunks, instead of fragments.

        // s = x * y
        Matrix<16, 32, u8> xa0, xa1, xa2, xa3;
        Matrix<32, 8, u8> yb;
        Matrix<16, 8, u32> sd0, sd1, sd2, sd3;
        Matrix<16, 8, u32> sbx;
        // u = m' * s (mod 2^256)
        Matrix<32, 8, u8> sb;
        Matrix<16, 32, u8> mp0, mp1;
        Matrix<16, 8, u32> ud0, ud1;
        Matrix<16, 8, u32> uds0, uds1; // ud after summing each two adjacent columns
        Matrix<16, 8, u32> ubx;
        // t = s + m * u
        Matrix<16, 32, u8> ma1, ma2, ma3;
        Matrix<32, 8, u8> ub;
        Matrix<16, 8, u32> td1, td2, td3;
        Matrix<16, 8, u32> tds1, tds2, tds3; // td after summing each two adjacent columns
        Matrix<16, 8, u32> tdc1, tdc2, tdc3; // tds after carrying up tail of first 32 u32's
        // z = t mod 2^256
        Matrix<16, 4, u32> tey;
        Matrix<16, 4, u16> rz; // t after compact
        Matrix<8, 4, u32> rw;
        Matrix<8, 4, u32> zw;

        Intermediates() = default;
        __host__ void alloc_device()
        {
          xa0.alloc_device();
          xa1.alloc_device();
          xa2.alloc_device();
          xa3.alloc_device();
          yb.alloc_device();
          sd0.alloc_device();
          sd1.alloc_device();
          sd2.alloc_device();
          sd3.alloc_device();
          sbx.alloc_device();
          sb.alloc_device();
          mp0.alloc_device();
          mp1.alloc_device();
          ud0.alloc_device();
          ud1.alloc_device();
          uds0.alloc_device();
          uds1.alloc_device();
          ubx.alloc_device();
          ma1.alloc_device();
          ma2.alloc_device();
          ma3.alloc_device();
          ub.alloc_device();
          td1.alloc_device();
          td2.alloc_device();
          td3.alloc_device();
          tds1.alloc_device();
          tds2.alloc_device();
          tds3.alloc_device();
          tdc1.alloc_device();
          tdc2.alloc_device();
          tdc3.alloc_device();
          tey.alloc_device();
          rz.alloc_device();
          rw.alloc_device();
          zw.alloc_device();
        }
        static __host__ Intermediates new_device()
        {
          Intermediates i;
          i.alloc_device();
          return i;
        }
        __host__ Intermediates to_host()
        {
          return Intermediates{
              .xa0 = xa0.to_host(),
              .xa1 = xa1.to_host(),
              .xa2 = xa2.to_host(),
              .xa3 = xa3.to_host(),
              .yb = yb.to_host(),
              .sd0 = sd0.to_host(),
              .sd1 = sd1.to_host(),
              .sd2 = sd2.to_host(),
              .sd3 = sd3.to_host(),
              .sbx = sbx.to_host(),
              .sb = sb.to_host(),
              .mp0 = mp0.to_host(),
              .mp1 = mp1.to_host(),
              .ud0 = ud0.to_host(),
              .ud1 = ud1.to_host(),
              .uds0 = uds0.to_host(),
              .uds1 = uds1.to_host(),
              .ubx = ubx.to_host(),
              .ma1 = ma1.to_host(),
              .ma2 = ma2.to_host(),
              .ma3 = ma3.to_host(),
              .ub = ub.to_host(),
              .td1 = td1.to_host(),
              .td2 = td2.to_host(),
              .td3 = td3.to_host(),
              .tds1 = tds1.to_host(),
              .tds2 = tds2.to_host(),
              .tds3 = tds3.to_host(),
              .tdc1 = tdc1.to_host(),
              .tdc2 = tdc2.to_host(),
              .tdc3 = tdc3.to_host(),
              .tey = tey.to_host(),
              .rz = rz.to_host(),
              .rw = rw.to_host(),
              .zw = zw.to_host()};
        }
      };

      std::ostream &operator<<(std::ostream &os, const Intermediates &i)
      {
        os << "xa0 = \n"
           << i.xa0;
        os << "xa1 = \n"
           << i.xa1;
        os << "xa2 = \n"
           << i.xa2;
        os << "xa3 = \n"
           << i.xa3;
        os << "yb = \n"
           << i.yb;
        os << "sd0 = \n"
           << i.sd0;
        os << "sd1 = \n"
           << i.sd1;
        os << "sd2 = \n"
           << i.sd2;
        os << "sd3 = \n"
           << i.sd3;
        os << "sbx = \n"
           << i.sbx;
        os << "sb = \n"
           << i.sb;
        os << "mp0 = \n"
           << i.mp0;
        os << "mp1 = \n"
           << i.mp1;
        os << "ud0 = \n"
           << i.ud0;
        os << "ud1 = \n"
           << i.ud1;
        os << "uds0 = \n"
           << i.uds0;
        os << "uds1 = \n"
           << i.uds1;
        os << "ubx = \n"
           << i.ubx;
        os << "ma1 = \n"
           << i.ma1;
        os << "ma2 = \n"
           << i.ma2;
        os << "ma3 = \n"
           << i.ma3;
        os << "ub = \n"
           << i.ub;
        os << "td1 = \n"
           << i.td1;
        os << "td2 = \n"
           << i.td2;
        os << "td3 = \n"
           << i.td3;
        os << "tds1 = \n"
           << i.tds1;
        os << "tds2 = \n"
           << i.tds2;
        os << "tds3 = \n"
           << i.tds3;
        os << "tdc1 = \n"
           << i.tdc1;
        os << "tdc2 = \n"
           << i.tdc2;
        os << "tdc3 = \n"
           << i.tdc3;
        os << "tey = \n"
           << i.tey;
        os << "rz = \n"
           << i.rz;
        os << "rw = \n"
           << i.rw;
        os << "zw = \n"
           << i.zw;
        return os;
      }

      // Put a A layotu matrix into memory
      __device__ __forceinline__ void store_a_matrix(
          u32 a0, u32 a1, u32 a2, u32 a3,
          Matrix<16, 32, u8> mat)
      {
        u32 lane_id = threadIdx.x % 32;

        auto st = [&mat, lane_id](u32 m, u32 n, u32 a)
        {
          u32 i = lane_id / 4 + m * 8;
          u32 j = (lane_id % 4) * 4 + n * 16;
          for (u32 k = 0; k < 4; k++)
            mat.get(i, j + k) = (a >> (8 * k)) & 0xff;
          // printf("[%u, %u] %u %u %u %u \n", i, j, mat.get(i, j + 0), mat.get(i, j + 1), mat.get(i, j + 2), mat.get(i, j + 3));
        };
        st(0, 0, a0);
        st(0, 1, a1);
        st(1, 0, a2);
        st(1, 1, a3);
      }

      // Put a B layotu matrix into memory
      __device__ __forceinline__ void store_b_matrix(
          u32 b0, u32 b1, Matrix<32, 8, u8> mat)
      {
        u32 lane_id = threadIdx.x % 32;

        auto st = [&mat, lane_id](u32 m, u32 b)
        {
          u32 i = (lane_id % 4) * 4 + m * 16;
          u32 j = lane_id / 4;
          for (u32 k = 0; k < 4; k++)
            mat.get(i + k, j) = (b >> (8 * k)) & 0xff;
        };

        st(0, b0);
        st(1, b1);
      }

      // Put a D layout matrix into memory
      __device__ __forceinline__ void store_d_matrix(
          u32 d0, u32 d1, u32 d2, u32 d3,
          Matrix<16, 8, u32> mat)
      {
        u32 lane_id = threadIdx.x % 32;

        auto st = [&mat, lane_id](u32 m, u32 n, u32 d)
        {
          u32 i = lane_id / 4 + m * 8;
          u32 j = (lane_id % 4) * 2 + n;
          mat.get(i, j) = d;
        };

        st(0, 0, d0);
        st(0, 1, d1);
        st(1, 0, d2);
        st(1, 1, d3);
      }

      // Put a Bx layout matrix into memory
      __device__ __forceinline__ void store_bx_matrix(
          u32 bx0, u32 bx1, u32 bx2, u32 bx3,
          Matrix<16, 8, u32> mat)
      {
        u32 lane_id = threadIdx.x % 32;

        auto st = [&mat, lane_id](u32 m, u32 bx)
        {
          u32 i = lane_id % 4 + m * 4;
          u32 j = lane_id / 4;
          mat.get(i, j) = bx;
        };

        st(0, bx0);
        st(1, bx1);
        st(2, bx2);
        st(3, bx3);
      }

      template <typename T = u16>
      __device__ __forceinline__ void store_z_matrix(
          T z0, T z1, Matrix<16, 4, T> mat)
      {
        u32 lane_id = threadIdx.x % 32;

        auto st = [&mat, lane_id](u32 m, u32 z)
        {
          u32 i = lane_id / 4 + 8 * m;
          u32 j = lane_id % 4;
          mat.get(i, j) = z;
        };

        st(0, z0);
        st(1, z1);
      }

      __device__ __forceinline__ void store_w_matrix(u32 w, Matrix<8, 4, u32> mat)
      {
        u32 lane_id = threadIdx.x % 32;
        u32 i = lane_id / 4;
        u32 j = lane_id % 4;
        mat.get(i, j) = w;
      }

      template <typename M>
      __device__ __forceinline__ void polulate_a_matrix(
          u32 &a0, u32 &a1, u32 &a2, u32 &a3, const M mat)
      {
        u32 lane_id = threadIdx.x % 32;

        auto ld = [mat, lane_id](u32 m, u32 n, u32 &a)
        {
          u32 i = lane_id / 4 + m * 8;
          u32 j = (lane_id % 4) * 4 + n * 16;
          for (u32 k = 0; k < 4; k++)
            a |= (mat(i, j + k)) << (8 * k);
        };
        ld(0, 0, a0);
        ld(0, 1, a1);
        ld(1, 0, a2);
        ld(1, 1, a3);
      }

      template <typename M>
      __device__ __forceinline__ void polulate_b_matrix(
          u32 &b0, u32 &b1, const M mat)
      {
        u32 lane_id = threadIdx.x % 32;

        auto ld = [mat, lane_id](u32 m, u32 &b)
        {
          u32 i = (lane_id % 4) * 4 + m * 16;
          u32 j = lane_id / 4;
          for (u32 k = 0; k < 4; k++)
            b |= (mat(i + k, j)) << (8 * k);
        };

        ld(0, b0);
        ld(1, b1);
      }

      template <typename M>
      __device__ __forceinline__ void polulate_d_matrix(
          u32 &d0, u32 &d1, u32 &d2, u32 &d3,
          const M mat)
      {
        u32 lane_id = threadIdx.x % 32;

        auto ld = [&mat, lane_id](u32 m, u32 n, u32 &d)
        {
          u32 i = lane_id / 4 + m * 8;
          u32 j = (lane_id % 4) * 2 + n;
          d = mat(i, j);
        };

        ld(0, 0, d0);
        ld(0, 1, d1);
        ld(1, 0, d2);
        ld(1, 1, d3);
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
      // Due to a mistake in reading documentation, I mistakenly swapped a1 and a2
      asm(
          "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};" : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(a0), "r"(a2), "r"(a1), "r"(a3),
                                                                                                                                                                           "r"(b0), "r"(b1),
                                                                                                                                                                           "r"(c0), "r"(c1), "r"(c2), "r"(c3));
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
    // This can be viewed as two matrices in BX layout.
    //
    // If mma is called with C = 0, then each D[ij] is at most 2^21, so the highest byte of D[ij] is zero.
    // This way, for example, D[0,0] and D[0,1] can be coalesced into a u32
    //   Bx[0,0] = D[0,0] + (D[0,1] << 8)
    // Note that Bx is not calculated in this function.
    __device__ __forceinline__ void transpose_d_to_b(
        u32 &b00, u32 &b01, u32 &b10, u32 &b11, u32 &b20, u32 &b21, u32 &b30, u32 &b31,
        u32 d00, u32 d02,
        u32 d10, u32 d12)
    {
      u32 lane_id = threadIdx.x % 32;
      u32 i = lane_id % 4;
      u32 j = lane_id / 4;
      u32 src_lane0 = 8 * i + j / 2;
      u32 src_lane1 = src_lane0 + 4;
      auto calc = [src_lane0, src_lane1](u32 &b0, u32 &b1, u32 d)
      {
        b0 = __shfl_sync(MASK_ALL, d, src_lane0);
        b1 = __shfl_sync(MASK_ALL, d, src_lane1);
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
    // The resulting By matrix is in B layout
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
      u32 mask = MASK_ALL;

      u16 bx0_hi, bx0_lo;
      u16 bx1_hi, bx1_lo;
      u16 bx2_hi, bx2_lo;
      u16 bx3_hi, bx3_lo;
      ptx::unpack_b32(bx0_lo, bx0_hi, bx0);
      ptx::unpack_b32(bx1_lo, bx1_hi, bx1);
      ptx::unpack_b32(bx2_lo, bx2_hi, bx2);
      ptx::unpack_b32(bx3_lo, bx3_hi, bx3);

      u32 src_var;

      by0 = 0;
      by1 = 0;

      // Round 1
      if (i % 2 == 0)
        src_var = bx0;
      else
        src_var = ptx::pack_b32(bx0_lo, bx1_lo);

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
        src_var = ptx::pack_b32(bx0_hi, bx1_hi);

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
        src_var = ptx::pack_b32(bx2_lo, bx3_lo);

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
        src_var = ptx::pack_b32(bx2_hi, bx3_hi);

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
    template <bool DEBUG>
    __device__ __forceinline__ void transpose_and_split(
        u32 &by0, u32 &by1,
        u32 d00, u32 d02,
        u32 d10, u32 d12,
        debug::Matrix<16, 8, u32> *bx = nullptr)
    {
      u32 db00, db01, db10, db11, db20, db21, db30, db31;
      transpose_d_to_b(
          db00, db01, db10, db11, db20, db21, db30, db31,
          d00, d02, d10, d12);
      u32 dbx0 = db00 + (db01 << 8);
      u32 dbx1 = db10 + (db11 << 8);
      u32 dbx2 = db20 + (db21 << 8);
      u32 dbx3 = db30 + (db31 << 8);
      if (DEBUG)
        debug::store_bx_matrix(dbx0, dbx1, dbx2, dbx3, *bx);
      shuffle_b_for_mul(
          by0, by1,
          dbx0, dbx1, dbx2, dbx3);
    }

    // Montgomery reduction zeros-out lower 256 bit of d = a * b, but the matrix column representing d consists of i32 elements,
    //   d = d[0] + d[1] * 2^8 + ... + d[30] * 2^240 + d[31] * 2^248 + d[32] * 2^256 + ... + d[63] * 2^504
    // so d[30]'s third byte (also the highest non-zero byte, d[30].2) has weight 2^256,
    // and d[31]'s second and third byte (d[31].1, d[31].2) has weight 2^256, 2^264.
    // These bytes must be added to d[32], which is what this function does.
    // However, this doesn't mean d[31].0, d[30].1 and all bytes with smaller weights are zero —— what are zero are sum of
    // those bytes with same weight. Thanks to this, we don't need to calculate d[28], d[27], ...
    //
    // Clearly, d[i].3's are all zeros, so for each weight, there are at most three bytes, and their sum is no bigger than
    //   3 * (2^8 - 1) = 3 * 2^8 - 3
    // By induction we can prove that the carry-out of this sum is no bigger than 3
    //   Sum with carry in <= 3 * (2^8 - 1) + 3 = 3 * 2^8
    // Based on this knowledge, we know for some x in {0, 1, 2, 3}
    //   d[29].2 + d[30].1 + d[31].0 + x = 0 (mod 2^8)
    // So the carry-out of sum at weight 2^248 equals to that of
    //   d[29].2 + d[30].1 + d[31].0 + 3
    // Directly adding this carry-out c to d[32] produces
    //   d mod 2^256 = (c + d[32]) * 2^256 + ... + d[63] * 2^504
    // After mma, lane 28's d12 holds d[30] and lane24's d12 holds d[31].
    __device__ __forceinline__ void carry_up_tail_of_d(
        u32 &d20, u32 d12)
    {
      u32 lane_id = threadIdx.x % 32;
      u32 i = lane_id / 4;
      u32 mask = MASK_ALL;
      u32 got = __shfl_sync(mask, d12, lane_id % 4 + 24);
      u32 s = 0;
      if (i == 0)
      {
        s += (got >> 8) & 0xff;
        d20 += got >> 16;
      }

      got = __shfl_sync(mask, d12, lane_id % 4 + 28);
      if (i == 0)
      {
        s += got & 0xff;
        d20 += got >> 8;
      }

      got = __shfl_sync(mask, d12, lane_id % 4 + 20);
      if (i == 0)
      {
        s += (got >> 16) & 0xff;
        s += 3;
        d20 += s >> 8;
      }
    }

    // Shuffle layout EX
    //
    //        0        1        2        3
    // 0  T0 {ex0}     T1       T2       T3
    // 1  T8           T9       T10      T11
    // 2  T16          T17      T18      T19
    // 3  T24          T25      T26      T27
    // 4  T0 {ex1}     T1       T2       T3
    // 5  T8           T9       T10      T11
    // 6  T16          T17      T18      T19
    // 7  T24          T25      T26      T27
    // 8  T0 {ex2}     T1       T2       T3
    // 9  T8           T9       T10      T11
    // 10 T16          T17      T18      T19
    // 11 T24          T25      T26      T27
    // 12 T0 {ex3}     T1       T2       T3
    // 13 T8           T9       T10      T11
    // 14 T16          T17      T18      T19
    // 15 T24          T25      T26      T27
    //
    // to layout EY
    //
    //        0        1        2        3
    // 0  T0 {ey0}     T1       T2       T3
    // 1  T4
    // 2  T8
    // 3  T12
    // 4  T16
    // 5  T20
    // 6  T24
    // 7  T28
    // 8  T0 {ey1}
    // 9  T4
    // 10 T8
    // 11 T12
    // 12 T16
    // 13 T20
    // 14 T24
    // 15 T28
    //
    __device__ __forceinline__ void shuffle_e_for_compact(
        u32 &ey0, u32 &ey1,
        u32 ex0, u32 ex1, u32 ex2, u32 ex3)
    {
      u32 lane_id = threadIdx.x % 32;
      u32 i = lane_id / 4;
      u32 j = lane_id % 4;
      u32 mask = MASK_ALL;

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

    template <bool IS_SUB = false, typename T, T T_MAX>
    __device__ __forceinline__ void fast_propagate_zw_template(
        T &r, u32 carry_out, u32 &carry_in)
    {
      u32 lane_id = threadIdx.x % 32;
      u32 i = lane_id / 4;
      u32 j = lane_id % 4;
      u32 mask = MASK_ALL;

      u32 saturated = IS_SUB ? r == 0 : r == T_MAX;

      u64 lco_b = __ballot_sync(mask, carry_out);
      u64 sat_b = __ballot_sync(mask, saturated);
      sat_b = (sat_b >> j) | 0xeeeeeeee;
      lco_b = (lco_b >> j) & 0x11111111;
      lco_b = (lco_b << 4) | carry_in;

      u64 carries = lco_b + sat_b;
      carry_in = carries >> 32;

      if ((carries ^ sat_b) & (1 << (lane_id - j)))
      {
        if (IS_SUB)
          r -= 1;
        else
          r += 1;
      }
    }

    template <bool IS_SUB = false>
    __device__ __forceinline__ void fast_propagate_z(
        u16 &r, u32 carry_out, u32 &carry_in)
    {
      fast_propagate_zw_template<IS_SUB, u16, 0xffff>(r, carry_out, carry_in);
    }

    template <bool IS_SUB = false>
    __device__ __forceinline__ void fast_propagate_w(
        u32 &r, u32 carry_out, u32 &carry_in)
    {
      fast_propagate_zw_template<IS_SUB, u32, 0xffffffff>(r, carry_out, carry_in);
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
    template <bool DEBUG>
    __device__ __forceinline__ void compact(
        u16 &r0, u16 &r1,
        u32 d00, u32 d02,
        u32 d10, u32 d12,
        debug::Matrix<16, 4, u32> *mey = nullptr)
    {
      u32 lane_id = threadIdx.x % 32;
      u32 i = lane_id / 4;
      u32 j = lane_id % 4;
      u32 mask = MASK_ALL;
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

      u32 ey0, ey1;
      shuffle_e_for_compact(ey0, ey1, e0, e1, e2, e3);

      if (i == 7)
      {
        ey1 = (ey1 & 0x0000ffff) | (ey0 & 0xffff0000);
        ey0 = ey0 & 0x0000ffff;
      }

      if (DEBUG)
        debug::store_z_matrix<u32>(ey0, ey1, *mey);

      u32 carry_in = 0;
      auto calc = [&carry_in, mask, lane_id, i, j](u16 &r, u32 ey)
      {
        u32 e_lower = __shfl_sync(mask, ey, lane_id - 4);
        u32 r_wide = (e_lower >> 16) + (ey & 0x0000ffff);
        r = r_wide & 0x0000ffff;
        u32 limb_carry_out = r_wide >> 16;

        fast_propagate_z(r, limb_carry_out, carry_in);
      };

      calc(r0, ey0);
      calc(r1, ey1);
    }

    __device__ __forceinline__ void add_w(
        u32 &r, u32 &carry_out, u32 a, u32 b)
    {
      r = ptx::add_cc(a, b);
      carry_out = 0;
      u32 limb_carry = ptx::addc(0, 0);
      fast_propagate_w<false>(r, limb_carry, carry_out);
    }

    __device__ __forceinline__ void sub_w(
        u32 &r, u32 &borrow_out, u32 a, u32 b)
    {
      r = ptx::sub_cc(a, b);
      borrow_out = 0;
      u32 limb_borrow = ptx::subc(0, 0);
      fast_propagate_w<true>(r, limb_borrow & 1, borrow_out);
    }

    // Compute z = r mod m.
    // z and r are both in W layout.
    __device__ __forceinline__ void modulo_m_w(
        u32 &z, u32 &r,
        const u32 *st_m)
    {
      u32 lane_id = threadIdx.x % 32;
      u32 i = lane_id / 4;
      u32 j = lane_id % 4;

      u32 borrow = 0;
      u32 useless;
      sub_w(z, borrow, r, st_m[i]);
      add_w(z, useless, z, borrow ? st_m[i] : 0);
    }

    // Shuffle Z layout to W layout
    //
    //          0       1        2        3
    //   0   T0{w0}     T1       T2       T3
    //   1   T4         T5       T6       T7
    //        ...
    //   7   T28        T29      T30      T31
    __device__ __forceinline__ void shuffle_z_to_w(
        u32 &w,
        u16 z0, u16 z1)
    {
      u32 lane_id = threadIdx.x % 32;
      u32 i = lane_id / 4;
      u32 j = lane_id % 4;
      u32 mask = MASK_ALL;

      w = 0;

      u32 send = ((u32)z1 << 16) + z0;
      u32 got = __shfl_sync(mask, send, (i % 4) * 8 + j);
      if (i < 4)
        w |= got & 0x0000ffff;
      else
        w |= got >> 16;

      got = __shfl_sync(mask, send, (i % 4) * 8 + 4 + j);
      if (i < 4)
        w |= got << 16;
      else
        w |= got & 0xffff0000;
    }

    // Store z to st_z, z is in Z layout.
    // `(MASK >> j) & 1` marks whether j-th column is to be stored to `st_w[j]`
    template <u32 MASK, typename F>
    __device__ __forceinline__ void store_w(
        F set, u32 w)
    {
      u32 lane_id = threadIdx.x % 32;
      u32 i = lane_id / 4;
      u32 j = lane_id % 4;

      if ((MASK >> j) & 1)
        set(j, i, w);
    }

    // Transmute four numbers in four columns from W layout to B layout
    __device__ __forceinline__ void transpose_w_to_b(
        u32 &b0, u32 &b1, u32 w)
    {
      u32 lane_id = threadIdx.x % 32;
      u32 j = lane_id / 8;
      u32 i = lane_id % 8;
      u32 src_lane = 4 * i + j;
      b0 = __shfl_sync(MASK_ALL, w, src_lane);
      b1 = __shfl_down_sync(MASK_ALL, b0, 4);
    }

    template <bool DEBUG = false>
    __device__ __forceinline__ void multiplication_raw(
        u32 &z,
        const u32 *st_x,
        u32 yb0, u32 yb1,
        debug::Intermediates *intermediates = nullptr
    )
    {

      if (DEBUG)
        debug::store_b_matrix(yb0, yb1, intermediates->yb);

      u32 xa0, xa1, xa2, xa3;

      load_a_matrix_shared(xa0, xa1, xa2, xa3, st_x, 8, 0);
      if (DEBUG)
        debug::store_a_matrix(xa0, xa1, xa2, xa3, intermediates->xa0);
      u32 sd00, sd01, sd02, sd03;
      mma_m16n8k32(sd00, sd01, sd02, sd03, xa0, xa1, xa2, xa3, yb0, yb1, 0, 0, 0, 0);

      load_a_matrix_shared(xa0, xa1, xa2, xa3, st_x, 8, 1);
      if (DEBUG)
        debug::store_a_matrix(xa0, xa1, xa2, xa3, intermediates->xa1);
      u32 sd10, sd11, sd12, sd13;
      mma_m16n8k32(sd10, sd11, sd12, sd13, xa0, xa1, xa2, xa3, yb0, yb1, 0, 0, 0, 0);

      load_a_matrix_shared(xa0, xa1, xa2, xa3, st_x, 8, 2);
      if (DEBUG)
        debug::store_a_matrix(xa0, xa1, xa2, xa3, intermediates->xa2);
      u32 sd20, sd21, sd22, sd23;
      mma_m16n8k32(sd20, sd21, sd22, sd23, xa0, xa1, xa2, xa3, yb0, yb1, 0, 0, 0, 0);

      load_a_matrix_shared(xa0, xa1, xa2, xa3, st_x, 8, 3);
      if (DEBUG)
        debug::store_a_matrix(xa0, xa1, xa2, xa3, intermediates->xa3);
      u32 sd30, sd31, sd32, sd33;
      mma_m16n8k32(sd30, sd31, sd32, sd33, xa0, xa1, xa2, xa3, yb0, yb1, 0, 0, 0, 0);

      if (DEBUG)
      {
        debug::store_d_matrix(sd00, sd01, sd02, sd03, intermediates->sd0);
        debug::store_d_matrix(sd10, sd11, sd12, sd13, intermediates->sd1);
        debug::store_d_matrix(sd20, sd21, sd22, sd23, intermediates->sd2);
        debug::store_d_matrix(sd30, sd31, sd32, sd33, intermediates->sd3);
      }

      u16 rz0, rz1;
      compact<DEBUG>(rz0, rz1, sd00, sd02, sd10, sd12, &intermediates->tey);

      if (DEBUG)
        debug::store_z_matrix(rz0, rz1, intermediates->rz);

      shuffle_z_to_w(z, rz0, rz1);
    }

    struct FragmentA
    {
      u32 a[24] = {0};
      FragmentA() = default;
      __device__ __forceinline__ void load(const u32 *p);
    };

    struct FragmentAForM
    {
      u32 m10, m11, m12, m13;
      u32 m20, m21, m22, m23;
      u32 m30, m31, m32, m33;

      __device__ __forceinline__ void load(FragmentA & fma);

    };

    struct FragmentAForMPrime
    {
      u32 mp00, mp01, mp02, mp03;
      u32 mp10, mp11, mp12, mp13;

      __device__ __forceinline__ void load(FragmentA& fmpa);
    };


    template <bool DEBUG = false>
    __device__ __forceinline__ void montgomery_multiplication_raw(
        u32 &z,
        const u32 *st_x,
        u32 yb0, u32 yb1,
        const FragmentAForM fam,
        const FragmentAForMPrime famp,
        const u32 *st_m,
        debug::Intermediates *intermediates = nullptr)
    {
      // Naming convention: <symbol> <layout> <number>
      // For example,        s        d        00
      //                     mp       a        0

      // Compute s = x * y

      if (DEBUG)
        debug::store_b_matrix(yb0, yb1, intermediates->yb);

      u32 xa0, xa1, xa2, xa3;

      load_a_matrix_shared(xa0, xa1, xa2, xa3, st_x, 8, 0);
      if (DEBUG)
        debug::store_a_matrix(xa0, xa1, xa2, xa3, intermediates->xa0);
      u32 sd00, sd01, sd02, sd03;
      mma_m16n8k32(sd00, sd01, sd02, sd03, xa0, xa1, xa2, xa3, yb0, yb1, 0, 0, 0, 0);

      load_a_matrix_shared(xa0, xa1, xa2, xa3, st_x, 8, 1);
      if (DEBUG)
        debug::store_a_matrix(xa0, xa1, xa2, xa3, intermediates->xa1);
      u32 sd10, sd11, sd12, sd13;
      mma_m16n8k32(sd10, sd11, sd12, sd13, xa0, xa1, xa2, xa3, yb0, yb1, 0, 0, 0, 0);

      load_a_matrix_shared(xa0, xa1, xa2, xa3, st_x, 8, 2);
      if (DEBUG)
        debug::store_a_matrix(xa0, xa1, xa2, xa3, intermediates->xa2);
      u32 sd20, sd21, sd22, sd23;
      mma_m16n8k32(sd20, sd21, sd22, sd23, xa0, xa1, xa2, xa3, yb0, yb1, 0, 0, 0, 0);

      load_a_matrix_shared(xa0, xa1, xa2, xa3, st_x, 8, 3);
      if (DEBUG)
        debug::store_a_matrix(xa0, xa1, xa2, xa3, intermediates->xa3);
      u32 sd30, sd31, sd32, sd33;
      mma_m16n8k32(sd30, sd31, sd32, sd33, xa0, xa1, xa2, xa3, yb0, yb1, 0, 0, 0, 0);

      if (DEBUG)
      {
        debug::store_d_matrix(sd00, sd01, sd02, sd03, intermediates->sd0);
        debug::store_d_matrix(sd10, sd11, sd12, sd13, intermediates->sd1);
        debug::store_d_matrix(sd20, sd21, sd22, sd23, intermediates->sd2);
        debug::store_d_matrix(sd30, sd31, sd32, sd33, intermediates->sd3);
      }

      // Compute u = m' * s (mod 2^256)

      u32 sb0, sb1;
      transpose_and_split<DEBUG>(sb0, sb1, sd00, sd02, sd10, sd12, &intermediates->sbx);
      if (DEBUG)
        debug::store_b_matrix(sb0, sb1, intermediates->sb);

      if (DEBUG)
        debug::store_a_matrix(famp.mp00, famp.mp01, famp.mp02, famp.mp03, intermediates->mp0);
      u32 ud00, ud01, ud02, ud03;
      mma_m16n8k32(ud00, ud01, ud02, ud03, famp.mp00, famp.mp01, famp.mp02, famp.mp03, sb0, sb1, 0, 0, 0, 0);

      if (DEBUG)
        debug::store_a_matrix(famp.mp10, famp.mp11, famp.mp12, famp.mp13, intermediates->mp1);
      u32 ud10, ud11, ud12, ud13;
      mma_m16n8k32(ud10, ud11, ud12, ud13, famp.mp10, famp.mp11, famp.mp12, famp.mp13, sb0, sb1, 0, 0, 0, 0);

      if (DEBUG)
      {
        debug::store_d_matrix(ud00, ud01, ud02, ud03, intermediates->ud0);
        debug::store_d_matrix(ud10, ud11, ud12, ud13, intermediates->ud1);
      }

      ud00 += ud01;
      ud02 += ud03;
      ud10 += ud11;
      ud12 += ud13;

      if (DEBUG)
      {
        debug::store_d_matrix(ud00, ud01, ud02, ud03, intermediates->uds0);
        debug::store_d_matrix(ud10, ud11, ud12, ud13, intermediates->uds1);
      }

      // Compute t = s + m * u

      u32 ub0, ub1;
      transpose_and_split<DEBUG>(ub0, ub1, ud00, ud02, ud10, ud12, &intermediates->ubx);
      if (DEBUG)
        debug::store_b_matrix(ub0, ub1, intermediates->ub);

      if (DEBUG)
        debug::store_a_matrix(fam.m10, fam.m11, fam.m12, fam.m13, intermediates->ma1);
      u32 td10, td11, td12, td13;
      mma_m16n8k32(td10, td11, td12, td13, fam.m10, fam.m11, fam.m12, fam.m13, ub0, ub1, sd10, 0, sd12, 0);

      if (DEBUG)
        debug::store_a_matrix(fam.m20, fam.m21, fam.m22, fam.m23, intermediates->ma2);
      u32 td20, td21, td22, td23;
      mma_m16n8k32(td20, td21, td22, td23,fam.m20, fam.m21, fam.m22, fam.m23, ub0, ub1, sd20, 0, sd22, 0);

      if (DEBUG)
        debug::store_a_matrix(fam.m30, fam.m31, fam.m32, fam.m33, intermediates->ma3);
      u32 td30, td31, td32, td33;
      mma_m16n8k32(td30, td31, td32, td33, fam.m30, fam.m31, fam.m32, fam.m33, ub0, ub1, sd30, 0, sd32, 0);

      if (DEBUG)
      {
        debug::store_d_matrix(td10, td11, td12, td13, intermediates->td1);
        debug::store_d_matrix(td20, td21, td22, td23, intermediates->td2);
        debug::store_d_matrix(td30, td31, td32, td33, intermediates->td3);
      }

      td10 += td11;
      td12 += td13;
      td20 += td21;
      td22 += td23;
      td30 += td31;
      td32 += td33;

      if (DEBUG)
      {
        debug::store_d_matrix(td10, td11, td12, td13, intermediates->tds1);
        debug::store_d_matrix(td20, td21, td22, td23, intermediates->tds2);
        debug::store_d_matrix(td30, td31, td32, td33, intermediates->tds3);
      }

      // Compute z = t mod 2^256

      carry_up_tail_of_d(td20, td12);

      if (DEBUG)
      {
        debug::store_d_matrix(td10, td11, td12, td13, intermediates->tdc1);
        debug::store_d_matrix(td20, td21, td22, td23, intermediates->tdc2);
        debug::store_d_matrix(td30, td31, td32, td33, intermediates->tdc3);
      }

      u16 rz0, rz1;
      compact<DEBUG>(rz0, rz1, td20, td22, td30, td32, &intermediates->tey);

      if (DEBUG)
        debug::store_z_matrix(rz0, rz1, intermediates->rz);

      u32 rw;
      shuffle_z_to_w(rw, rz0, rz1);

      if (DEBUG)
        debug::store_w_matrix(rw, intermediates->rw);

      modulo_m_w(z, rw, st_m);

      if (DEBUG)
        debug::store_w_matrix(z, intermediates->zw);
    }

    __device__ __forceinline__ void FragmentA::load(const u32 *p)
    {
      u32 lane_id = threadIdx.x % 32;
      bool predicate = (lane_id >= 8) && (lane_id < 16);
      a[lane_id] = predicate ? p[lane_id - 8] : 0;
    }

    struct FragmentB
    {
      u32 b0, b1;

      template <u32 MASK, typename F>
      static __device__ __forceinline__ FragmentB load(F p)
      {
        FragmentB fb;
        load_b_matrix<MASK>(fb.b0, fb.b1, p);
        return fb;
      }
    };

    struct FragmentW
    {
      u32 w;
      template <u32 MASK, typename F>
      __device__ __forceinline__ void store(F set)
      {
        store_w<MASK>(set, w);
      }

      template <u32 PERMUTED>
      __device__ __forceinline__ void transmute_columns()
      {
        u32 lane_id = threadIdx.x % 32;
        u32 j = lane_id % 4;
        u32 src_column = (PERMUTED >> (j * 2)) & 0b11;
        u32 src_lane = lane_id - j + src_column;
        w = __shfl_sync(MASK_ALL, w, src_lane);
      }

      __device__ __forceinline__ FragmentB transpose_to_b()
      {
        FragmentB fb;
        transpose_w_to_b(fb.b0, fb.b1, w);
        return fb;
      }
    };

    template <typename Params>
    struct ConstantLoader
    {
      FragmentA m, m_prime;
      ConstantLoader() = default;
      __device__ __forceinline__ void load()
      {
        m.load(Params::m().limbs);
        m_prime.load(Params::m_prime_wide().limbs);
      }
    };

    __device__ __forceinline__ void FragmentAForM::load(FragmentA & fma)
          {
      load_a_matrix_shared(m10, m11, m12, m13, fma.a, 8, 1);
      load_a_matrix_shared(m20, m21, m22, m23, fma.a, 8, 2);
      load_a_matrix_shared(m30, m31, m32, m33, fma.a, 8, 3);
    }

    __device__ __forceinline__ void FragmentAForMPrime::load(FragmentA& fmpa)
    {
      load_a_matrix_shared(mp00, mp01, mp02, mp03, fmpa.a, 8, 0);
      load_a_matrix_shared(mp10, mp11, mp12, mp13, fmpa.a, 8, 1);
    }

    template <typename Params>
    struct Multiplier
    {
      FragmentAForM m;
      FragmentAForMPrime m_prime;

      __device__ Multiplier(ConstantLoader<Params> &cl) 
      {
        m.load(cl.m);
        m_prime.load(cl.m_prime);
      }

      template <bool DEBUG = false>
      __device__ __forceinline__ FragmentW execute(FragmentA& x, FragmentB& y, debug::Intermediates *i = nullptr)
      {
        FragmentW w;
        montgomery_multiplication_raw<DEBUG>(
            w.w, x.a, y.b0, y.b1, m, m_prime, Params::m().limbs, i);
        return w;
      }

      __device__ __forceinline__ FragmentW operator()(FragmentA &x, FragmentB& y)
      {
        return execute(x, y);
      }
    };

    template <bool DEBUG = false>
    __device__ __forceinline__ FragmentW number_multiplication(FragmentA x, FragmentB y, debug::Intermediates *i = nullptr)
    {
      FragmentW w;
      multiplication_raw<DEBUG>(w.w, x.a, y.b0, y.b1, i);
      return w;
    }

  }
}

#endif