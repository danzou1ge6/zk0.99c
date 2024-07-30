#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <iostream>

#include "../../mont/src/mont.cuh"
#include "../src/bn254.cuh"

using namespace mont256;
using namespace curve256;

__global__ void to_affine_kernel(PointAffine *pr, const Point *p)
{
  auto cur = bn254::new_bn254();
  *pr = cur.to_affine(*p);
}

__global__ void from_affine_kernel(Point *pr, const PointAffine *p)
{
  auto cur = bn254::new_bn254();
  *pr = cur.from_affine(*p);
}

__global__ void is_on_curve_kernel(bool *r, const Point *p)
{
  auto cur = bn254::new_bn254();
  *r = cur.is_on_curve(*p);
}

__global__ void is_on_curve_kernel(bool *r, const PointAffine *p)
{
  auto cur = bn254::new_bn254();
  *r = cur.is_on_curve(*p);
}

__global__ void self_add_kernel(Point *pr, const Point *p)
{
  auto cur = bn254::new_bn254();
  *pr = cur.self_add(*p);
}

__global__ void add_kernel(Point *pr, const Point *pa, const Point *pb)
{
  auto cur = bn254::new_bn254();
  *pr = cur.add(*pa, *pb);
}

__global__ void add_kernel(Point *pr, const Point *pa, const PointAffine *pb)
{
  auto cur = bn254::new_bn254();
  *pr = cur.add(*pa, *pb);
}

__global__ void eq_kernel(bool *r, const Point *pa, const Point *pb)
{
  auto cur = bn254::new_bn254();
  *r = cur.eq(*pa, *pb);
}

__global__ void eq_kernel(bool *r, const PointAffine *pa, const PointAffine *pb)
{
  auto cur = bn254::new_bn254();
  *r = cur.eq(*pa, *pb);
}

__global__ void multiple_kernel(Point *r, const Point *p, u32 n)
{
  auto cur = bn254::new_bn254();
  *r = cur.multiple(p, n);
}

void to_affine(PointAffine *pr, const Point *p)
{
  to_affine_kernel<<<1, 1>>>(pr, p);
}

void from_affine(Point *pr, const PointAffine *p)
{
  from_affine_kernel<<<1, 1>>>(pr, p);
}

void is_on_curve(bool *r, const Point *p)
{
  is_on_curve_kernel<<<1, 1>>>(r, p);
}

void is_on_curve(bool *r, const PointAffine *p)
{
  is_on_curve_kernel<<<1, 1>>>(r, p);
}

void self_add(Point *pr, const Point *p)
{
  self_add_kernel<<<1, 1>>>(pr, p);
}

void add(Point *pr, const Point *pa, const Point *pb)
{
  add_kernel<<<1, 1>>>(pr, pa, pb);
}

void add(Point *pr, const Point *pa, const PointAffine *pb)
{
  add_kernel<<<1, 1>>>(pr, pa, pb);
}

void eq(bool *r, const Point *pa, const Point *pb)
{
  eq_kernel<<<1, 1>>>(r, pa, pb);
}

void eq(bool *r, const PointAffine *pa, const PointAffine *pb)
{
  eq_kernel<<<1, 1>>>(r, pa, pb);
}

void multiple(Point *r, const Point *p, u32 n)
{
  multiple_kernel<<<1, 1>>>(r, p, n);
}

template <typename R, typename T>
R launch_kernel1(T &a, void kernel(R *r, const T *a))
{
  R *dr;
  T *dt;
  cudaMalloc(&dt, sizeof(T));
  cudaMalloc(&dr, sizeof(R));
  cudaMemcpy(dt, &a, sizeof(T), cudaMemcpyHostToDevice);

  auto err = cudaGetLastError();
  if (err != cudaSuccess)
    FAIL("CUDA error: ", cudaGetErrorString(err));

  kernel(dr, dt);

  if (err != cudaSuccess)
    FAIL("CUDA error: ", cudaGetErrorString(err));

  R r;
  cudaMemcpy(&r, dr, sizeof(R), cudaMemcpyDeviceToHost);
  return r;
}

template <typename R, typename T1, typename T2>
R launch_kernel2(T1 &a, T2 &b, void kernel(R *r, const T1 *t1, const T2 *t2))
{
  R *dr;
  T1 *dt1;
  T2 *dt2;
  cudaMalloc(&dt1, sizeof(T1));
  cudaMalloc(&dt2, sizeof(T2));
  cudaMalloc(&dr, sizeof(R));
  cudaMemcpy(dt1, &a, sizeof(T1), cudaMemcpyHostToDevice);
  cudaMemcpy(dt2, &b, sizeof(T2), cudaMemcpyHostToDevice);

  auto err = cudaGetLastError();
  if (err != cudaSuccess)
    FAIL("CUDA error: ", cudaGetErrorString(err));

  kernel(dr, dt1, dt2);

  if (err != cudaSuccess)
    FAIL("CUDA error: ", cudaGetErrorString(err));

  R r;
  cudaMemcpy(&r, dr, sizeof(R), cudaMemcpyDeviceToHost);
  return r;
}

// All in Montgomery representation
// x = 17588249314949534242365104770887097184252481281543984788935084765326940834124
// x_mont = 15118613892300968952598744084540872064370112947361953331248357527216742275022
const u32 p1_x[8] = BIG_INTEGER_CHUNKS8(0x216cd50c, 0x64543f25, 0x23ea3cd2, 0xb993a6dd, 0x83f1909e, 0xfa432c22, 0xb05c2787, 0x423517ce);
// y = 12357679330807769286224777493623770042961632047811765801488126600704678244454
// y_mont = 15153950746392142037506258921283735096067572820412603689246231231476536618320
const u32 p1_y[8] = BIG_INTEGER_CHUNKS8(0x2180d509, 0x28461b27, 0xa030a1a7, 0x99911e11, 0xba9ce74f, 0xb488be66, 0x48fee199, 0x2592ad50);
// x = 12608279712251949873380244143875219782586201208786743136308610799954805067194
// x_mont = 19167055408201548293887119995526340569960432486177240658994856871947018094871
const u32 p2_x[8] = BIG_INTEGER_CHUNKS8(0x2a602b3e, 0x1b4f3db1, 0xbe03a8be, 0x55ed2444, 0xd8128f84, 0x59d8a382, 0x20aca79d, 0xf58ac917);
// y = 16073978196211952434062052344575429549195455908288056916460411561436103132293
// y_mont = 14394749309957023594572365872892788643083852397597524596381726046325896367924
const u32 p2_y[8] = BIG_INTEGER_CHUNKS8(0x1fd323ae, 0xc7ed1080, 0xf3890048, 0x493e9f2f, 0x243c445d, 0xd7b99618, 0xaefe84ff, 0xa1dbcf34);

PointAffine load_affine(const u32 x_data[8], const u32 y_data[8])
{
  auto x = Element::load(x_data);
  auto y = Element::load(y_data);
  return PointAffine(x, y);
}

void test_affine_projective_back_and_forth(const u32 x_data[8], const u32 y_data[8])
{
  auto affine = load_affine(x_data, y_data);

  auto projective = launch_kernel1(affine, from_affine);
  auto affine2 = launch_kernel1(projective, to_affine);
  auto projective2 = launch_kernel1(affine2, from_affine);

  REQUIRE(launch_kernel1(projective, is_on_curve));
  REQUIRE(launch_kernel1(projective2, is_on_curve));
  REQUIRE(launch_kernel1(affine2, is_on_curve));

  REQUIRE(launch_kernel2(affine, affine2, eq));
  REQUIRE(launch_kernel2(projective, projective2, eq));
}

TEST_CASE("On curve")
{
  auto p1 = load_affine(p1_x, p1_y);
  auto p2 = load_affine(p2_x, p2_y);

  REQUIRE(launch_kernel1(p1, is_on_curve));
  REQUIRE(launch_kernel1(p2, is_on_curve));
}

TEST_CASE("Affine/Projective back and forth 1")
{
  test_affine_projective_back_and_forth(p1_x, p1_y);
}

TEST_CASE("Affine/Projective back and forth 2")
{
  test_affine_projective_back_and_forth(p2_x, p2_y);
}

TEST_CASE("Point addition commutative")
{
  auto p1 = load_affine(p1_x, p1_y);
  auto p2 = load_affine(p2_x, p2_y);
  auto p1p = launch_kernel1(p1, from_affine);
  auto p2p = launch_kernel1(p2, from_affine);

  auto sum1 = launch_kernel2(p1p, p2p, add);
  auto sum2 = launch_kernel2(p2p, p1p, add);

  REQUIRE(launch_kernel1(sum1, is_on_curve));
  REQUIRE(launch_kernel1(sum2, is_on_curve));

  REQUIRE(launch_kernel2(sum1, sum2, eq));
}

TEST_CASE("Point-PointAffine addition equivalent")
{
  auto p1 = load_affine(p1_x, p1_y);
  auto p2 = load_affine(p2_x, p2_y);
  auto p1p = launch_kernel1(p1, from_affine);
  auto p2p = launch_kernel1(p2, from_affine);

  auto sum1 = launch_kernel2(p2p, p1, add);
  auto sum2 = launch_kernel2(p2p, p1p, add);

  REQUIRE(launch_kernel1(sum1, is_on_curve));
  REQUIRE(launch_kernel1(sum2, is_on_curve));

  REQUIRE(launch_kernel2(sum1, sum2, eq));
}

TEST_CASE("Doubling equivalent")
{
  auto p1 = load_affine(p1_x, p1_y);
  auto p1p = launch_kernel1(p1, from_affine);

  auto sum1 = launch_kernel2(p1p, p1p, add);
  auto sum2 = launch_kernel1(p1p, self_add);

  REQUIRE(launch_kernel1(sum1, is_on_curve));
  REQUIRE(launch_kernel1(sum2, is_on_curve));

  REQUIRE(launch_kernel2(sum1, sum2, eq));
}