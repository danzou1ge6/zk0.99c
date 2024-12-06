#include "../src/bn254.cuh"
#include "../src/msm_radix_sort.cuh"
#include "../../mont/src/bn254_scalar.cuh"

#include <iostream>
#include <fstream>

using bn254::Point;
using bn254::PointAffine;
using bn254_scalar::Element;
using bn254_scalar::Number;
using mont::u32;
using mont::u64;

struct MsmProblem
{
  u32 len;
  PointAffine *points;
  Element *scalers;
};

std::istream &
operator>>(std::istream &is, MsmProblem &msm)
{
  is >> msm.len;
  msm.scalers = new Element[msm.len];
  msm.points = new PointAffine[msm.len];
  for (u32 i = 0; i < msm.len; i++)
  {
    char _;
    is >> msm.scalers[i].n >> _ >> msm.points[i];
  }
  return is;
}

std::ostream &
operator<<(std::ostream &os, const MsmProblem &msm)
{

  for (u32 i = 0; i < msm.len; i++)
  {
    os << msm.scalers[i].n << '|' << msm.points[i] << std::endl;
  }
  return os;
}

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    std::cout << "usage: <prog> input_file" << std::endl;
    return 2;
  }

  std::ifstream rf(argv[1]);
  if (!rf.is_open())
  {
    std::cout << "open file " << argv[1] << " failed" << std::endl;
    return 3;
  }

  MsmProblem msm;

  rf >> msm;

  cudaHostRegister((void*)msm.scalers, msm.len * sizeof(Element), cudaHostRegisterDefault);
  cudaHostRegister((void*)msm.points, msm.len * sizeof(PointAffine), cudaHostRegisterDefault);

  u32 *d_points, *h_points_precompute, head;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  msm::precompute<msm::MsmConfig<>>((u32*)msm.points, msm.len, d_points, h_points_precompute, head, stream);

  cudaEvent_t start, stop;
  float elapsedTime = 0.0;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  Point r;
  msm::run<msm::MsmConfig<>>((u32*)msm.scalers, d_points, msm.len, r, h_points_precompute, head, stream);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);

  cudaFree(d_points);

  std::cout << r.to_affine() << std::endl;

  std::cout << "Total cost time:" << elapsedTime << std::endl;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaHostUnregister((void*)msm.scalers);
  cudaHostUnregister((void*)msm.points);
  cudaFreeHost(h_points_precompute);
  cudaFree(d_points);

  return 0;
}