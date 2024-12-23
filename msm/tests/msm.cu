#include "../src/bn254.cuh"
#include "../src/msm.cuh"
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

  using Config = msm::MsmConfig<255, 22, 2, false>;
  u32 batch_size = 4;
  u32 batch_per_run = 2;
  u32 parts = 8;
  u32 stage_scalers = 2;
  u32 stage_points = 2;

  std::array<u32*, Config::n_precompute> h_points;
  h_points[0] = (u32*)msm.points;
  for (u32 i = 1; i < Config::n_precompute; i++) {
    cudaHostAlloc(&h_points[i], msm.len * sizeof(PointAffine), cudaHostAllocDefault);
  }

  
  std::vector<u32*> scalers_batches;
  for (int i = 0; i < batch_size; i++) {
    scalers_batches.push_back((u32*)msm.scalers);
  }

  std::vector<Point> r(batch_size);

  std::vector<u32> cards;
  int card_count;
  cudaGetDeviceCount(&card_count);
  for (int i = 0; i < card_count; i++) {
    cards.push_back(i);
  }

  msm::MultiGPUMSM<Config, Number, Point, PointAffine> msm_solver(msm.len, batch_per_run, parts, stage_scalers, stage_points, cards);

  std::cout << "start precompute" << std::endl;

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  msm::MSMPrecompute<Config, Point, PointAffine>::precompute(msm.len, h_points, 4);
  msm_solver.set_points(h_points);

  std::cout << "Precompute done" << std::endl;
  msm_solver.alloc_gpu();
  std::cout << "Alloc GPU done" << std::endl;
  cudaEvent_t start, stop;
  float elapsedTime = 0.0;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  msm_solver.msm(scalers_batches, r);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  std::cout << "Run done" << std::endl;

  cudaStreamDestroy(stream);

  for (int i = 0; i < batch_size; i++) {
    std::cout << r[i].to_affine() << std::endl;
  }

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