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

  u32 *d_points;
  bool head;

  using Config = msm::MsmConfig<>;
  u32 batch_size = 2;
  u32 parts = 8;
  u32 stage_scalers = 2;
  u32 stage_points = 2;

  u32 *h_points[Config::n_precompute];
  h_points[0] = (u32*)msm.points;
  for (u32 i = 1; i < Config::n_precompute; i++) {
    cudaHostAlloc(&h_points[i], msm.len * sizeof(PointAffine), cudaHostAllocDefault);
  }

  const u32 **scalers = new const u32*[batch_size];
  scalers[0] = (u32*)msm.scalers;

  for (int i = 1; i < batch_size; i++) {
    cudaHostAlloc(&scalers[i], msm.len * sizeof(Element), cudaHostAllocDefault);
    memcpy((void*)scalers[i], (void*)msm.scalers, msm.len * sizeof(Element));
  }
  Point *r = new Point[batch_size];

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  msm::precompute<Config>(h_points, msm.len, stream);

  cudaEvent_t start, stop;
  float elapsedTime = 0.0;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  msm::run<Config>(msm.len, batch_size, parts, stage_scalers, stage_points, scalers, const_cast<const u32 **>(h_points), r, false, false, d_points, head, stream);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);

  for (int i = 1; i < batch_size; i++) {
    cudaFreeHost((void*)scalers[i]);
  }

  delete [] scalers;

  cudaStreamDestroy(stream);

  for (int i = 0; i < batch_size; i++) {
    std::cout << r[i].to_affine() << std::endl;
  }

  delete [] r;

  std::cout << "Total cost time:" << elapsedTime << std::endl;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaHostUnregister((void*)msm.scalers);
  cudaHostUnregister((void*)msm.points);
  for (u32 i = 1; i < Config::n_precompute; i++) {
    cudaFreeHost(h_points[i]);
  }

  return 0;
}