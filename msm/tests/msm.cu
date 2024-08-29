#include "../src/curve.cuh"
#include "../src/bn254.cuh"
#include "../src/msm.cuh"
#include "../../mont/src/mont.cuh"

#include <iostream>

using namespace mont256;
using namespace curve256;

struct MsmProblem
{
  u32 len;
  u32 *scalers, *points;
};

std::istream &
operator>>(std::istream &is, MsmProblem &msm)
{
  is >> msm.len;
  msm.scalers = new u32[msm.len * Number::N_WORDS];
  msm.points = new u32[msm.len * PointAffine::N_WORDS];
  for (u32 i = 0; i < msm.len; i++)
  {
    Element scaler;
    PointAffine point;
    char _;
    is >> scaler >> _ >> point;
    scaler.store(msm.scalers + i * Number::N_WORDS);
    point.store(msm.points + i * PointAffine::N_WORDS);
  }
  return is;
}

std::ostream &
operator<<(std::ostream &os, const MsmProblem &msm)
{

  for (u32 i = 0; i < msm.len; i++)
  {
    auto scaler = Element::load(msm.scalers + i * Number::N_WORDS);
    auto point = PointAffine::load(msm.points + i * PointAffine::N_WORDS);
    os << scaler << '|' << point << std::endl;
  }
  return os;
}

int main()
{
  MsmProblem msm;
  std:: cin >> msm;
  Point r;
  msm::run<msm::MsmConfig>(msm.scalers, msm.points, msm.len, r);
  std::cout << r;
  return 0;
}


// __global__ void warp_reduction(u32 *points, u32 len, Point *reduced)
// {
//   auto curve = bn254::new_bn254();
//   using WarpReduce = cub::WarpReduce<Point>;
//   __shared__ typename WarpReduce::TempStorage temp_storage[2];
//   auto p = PointAffine::load(points + PointAffine::N_WORDS * (threadIdx.x % len));
//   Point reduced0 = WarpReduce(temp_storage[threadIdx.x / 32]).Reduce(curve.from_affine(p), [&curve](const Point &a, const Point &b)
//                                                   { return curve.add(a, b); });
//   if (threadIdx.x == 0)
//     *reduced = reduced0;
//   // PointAffine p1 = PointAffine::load(points);
//   // PointAffine p2 = PointAffine::load(points + PointAffine::N_WORDS);

//   // printf("on curve = %u\n", curve.is_on_curve(p1));
//   // printf("on curve = %u\n", curve.is_on_curve(p2));

//   // *reduced = curve.add(curve.from_affine(p1), p2);

//   // printf("on curve = %u\n", curve.is_on_curve(*reduced));
// }

// int main()
// {
//   MsmProblem msm;
//   std::cin >> msm;
//   u32* d_points;
//   cudaMalloc(&d_points, sizeof(u32) * PointAffine::N_WORDS * msm.len);
//   cudaMemcpy(d_points, msm.points, sizeof(u32) * PointAffine::N_WORDS * msm.len, cudaMemcpyHostToDevice);
//   Point* d_reduced;
//   cudaMalloc(&d_reduced, sizeof(Point));
//   warp_reduction<<<1, 64>>>(d_points, msm.len, d_reduced);

//   cudaError_t err = cudaGetLastError();
//   if (err != cudaSuccess)
//     std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;

//   Point reduced;
//   cudaMemcpy(&reduced, d_reduced, sizeof(Point), cudaMemcpyDeviceToHost);
//   // std::cout << PointAffine::load(msm.points) << std::endl;
//   // std::cout << PointAffine::load(msm.points + PointAffine::N_WORDS) << std::endl;
//   std::cout << reduced;
//   return 0;
// }

