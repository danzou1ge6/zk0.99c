#include "../src/bn254.cuh"
#include "../src/msm.cuh"

#include <iostream>
#include <fstream>

using bn254::Point;
using bn254::PointAffine;
using bn254::PointAll;
using bn254::PointAffineAll;
using Number = mont::Number<8>;
using mont::u32;
using mont::u64;

#ifndef WINDOW_S
#define WINDOW_S 16
#endif

#ifndef ALPHA
#define ALPHA 16
#endif

struct MsmProblem
{
  u32 len;
  PointAffine *points;
  Number *scalers;
};

std::istream &
operator>>(std::istream &is, MsmProblem &msm)
{
  is >> msm.len;
  msm.scalers = new Number[msm.len];
  msm.points = new PointAffine[msm.len * TPI];
  for (u32 i = 0; i < msm.len; i++)
  {
    char _;
    PointAffineAll p;
    is >> msm.scalers[i] >> _ >> p;
    for(int j=0; j<TPI; ++j) {
      int PER_LIMBS = PointAffine::N_WORDS / 2;
      for(int k=0; k<PER_LIMBS; k++) {
        msm.points[i*TPI+j].x.n._limbs[k] = p.x.n._limbs[j*PER_LIMBS+k];
        msm.points[i*TPI+j].y.n._limbs[k] = p.y.n._limbs[j*PER_LIMBS+k];
      }
    }
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
  cudaHostRegister((void*)msm.points, msm.len * sizeof(PointAffine) * TPI, cudaHostRegisterDefault);

  using Config = msm::MsmConfig<255, WINDOW_S, ALPHA, false>;
  u32 batch_size = 1;
  u32 batch_per_run;
  u32 parts;
  u32 stage_scalers = 2;
  u32 stage_points = 2;

  std::array<u32*, Config::n_precompute> h_points;
  h_points[0] = (u32*)msm.points;
  for (u32 i = 1; i < Config::n_precompute; i++) {
    cudaHostAlloc(&h_points[i], msm.len * sizeof(PointAffine) * TPI, cudaHostAllocDefault);
  }

  
  std::vector<u32*> scalers_batches;
  for (int i = 0; i < batch_size; i++) {
    scalers_batches.push_back((u32*)msm.scalers);
  }

  std::vector<PointAll> r(batch_size);

  std::vector<u32> cards;
  int card_count;
  cudaGetDeviceCount(&card_count);
  for (int i = 0; i < card_count; i++) {
    cards.push_back(i);
  }

  for(batch_per_run = 1; batch_per_run <= batch_size; batch_per_run *= 2)
  {
    for(parts = 2; parts <= 2; parts *= 2)
    {
      msm::MultiGPUMSM<Config, Number, Point, PointAffine, PointAll> msm_solver(msm.len, batch_per_run, parts, stage_scalers, stage_points, cards);

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

      // std::cout << "Stage1" << std::endl;
      msm_solver.msm(scalers_batches, r);
      // printf("Stage16\n");

      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&elapsedTime, start, stop);
      std::cout << "Run done" << std::endl;

      cudaStreamDestroy(stream);

      for (int i = 0; i < batch_size; i++) {
        std::cout << r[i].to_affine() << std::endl;
      }

      std::cout << "window_size:0x" << Config::s << " alpha:0x" << Config::n_windows << " parts:0x" << parts << " batchs_per_run:0x" << batch_per_run << std::endl;
      std::cout << "Total cost time:" << elapsedTime << std::endl;
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
    }
  }
  

  cudaHostUnregister((void*)msm.scalers);
  cudaHostUnregister((void*)msm.points);
  for (u32 i = 1; i < Config::n_precompute; i++) {
    cudaFreeHost(h_points[i]);
  }

  return 0;
}