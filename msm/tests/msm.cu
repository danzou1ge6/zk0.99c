#include <iostream>
#include <fstream>

#include "../src/msm.cuh"

#if defined(CURVE_BN254)
#include "../src/bn254.cuh"
using bn254::Point;
using bn254::PointAffine;
using bn254::PointAll;
using bn254::PointAffineAll;
using Number = mont::Number<8>;

#elif defined(CURVE_BLS12381)
#include "../src/bls12381.cuh"
using bls12381::Point;
using bls12381::PointAffine;
using bls12381::PointAll;
using bls12381::PointAffineAll;
using Number = mont::Number<8>;

#elif defined(CURVE_MNT4753)
#include "../src/mnt4753.cuh"
using mnt4753::Point;
using mnt4753::PointAffine;
using mnt4753::PointAll;
using mnt4753::PointAffineAll;
using Number = mont::Number<24>;
#endif

using mont::u32;
using mont::u64;

#ifndef WINDOW_S
#define WINDOW_S 16
#endif

#ifndef ALPHA
#define ALPHA 16
#endif

#ifndef BATCH_SIZE
#define BATCH_SIZE 1
#endif

#ifndef BATCH_PER_RUN
#define BATCH_PER_RUN 1
#endif

#ifndef PARTS
#define PARTS 2
#endif

struct MsmProblem
{
  u64 len;
  PointAffineAll *points;
  Number *scalers;
};

std::istream &
operator>>(std::istream &is, MsmProblem &msm)
{
  is >> msm.len;
  msm.scalers = new Number[msm.len];
  msm.points = new PointAffineAll[msm.len];
  for (u32 i = 0; i < msm.len; i++)
  {
    char _;
    is >> msm.scalers[i] >> _ >> msm.points[i];
  }
  return is;
}

// std::ostream &
// operator<<(std::ostream &os, const MsmProblem &msm)
// {

//   for (u32 i = 0; i < msm.len; i++)
//   {
//     os << msm.scalers[i].n << '|' << msm.points[i] << std::endl;
//   }
//   return os;
// }

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

  cudaHostRegister((void*)msm.scalers, msm.len * sizeof(Number), cudaHostRegisterDefault);
  cudaHostRegister((void*)msm.points, msm.len * sizeof(PointAffineAll), cudaHostRegisterDefault);
#if defined(CURVE_BN254)
  using Config = msm::MsmConfig<255, WINDOW_S, ALPHA, false, TPI>;
#elif defined(CURVE_BLS12381)
  using Config = msm::MsmConfig<255, WINDOW_S, ALPHA, false, TPI>;
#elif defined(CURVE_MNT4753)
  using Config = msm::MsmConfig<753, WINDOW_S, ALPHA, false, TPI>;
#endif
  u32 stage_scalers = 2;
  u32 stage_points = 2;

  std::array<u32*, Config::n_precompute> h_points;
  h_points[0] = (u32*)msm.points;
  for (u32 i = 1; i < Config::n_precompute; i++) {
    cudaHostAlloc(&h_points[i], msm.len * sizeof(PointAffineAll), cudaHostAllocDefault);
  }

  
  std::vector<u32*> scalers_batches;
  for (int i = 0; i < BATCH_SIZE; i++) {
    scalers_batches.push_back((u32*)msm.scalers);
  }

  std::vector<PointAll> r(BATCH_SIZE);

  std::vector<u32> cards;
  int card_count;
  cudaGetDeviceCount(&card_count);
  for (int i = 0; i < card_count; i++) {
    cards.push_back(i);
  }

  msm::MultiGPUMSM<Config, Number, Point, PointAffine, PointAll, PointAffineAll> msm_solver(msm.len, BATCH_PER_RUN, PARTS, stage_scalers, stage_points, cards);

  std::cout << "start precompute" << std::endl;

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  msm::MSMPrecompute<Config, Point, PointAffine, PointAffineAll>::precompute(msm.len, h_points, 4);
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

  for (int i = 0; i < BATCH_SIZE; i++) {
    std::cout << r[i].to_affine() << std::endl;
  }

  std::cout << "window_size:0x" << Config::s << " alpha:0x" << Config::n_windows << " parts:0x" << PARTS << " batchs_per_run:0x" << BATCH_PER_RUN << std::endl;
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