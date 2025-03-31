#include "../src/alt_bn128.cuh"
#include "../src/msm.cuh"
#include "../../mont/src/alt_bn128_fr.cuh"

#include <iostream>
#include <fstream>

using mont::u32;
using mont::u64;

template <typename G>
struct MsmPoints
{
  u32 len;
  G *ptr;
};

template <typename G>
std::istream &
operator>>(std::istream &is, MsmPoints<G> &points)
{
  is >> points.len;
  std::cout << "Points len = " << points.len << std::endl;
  points.ptr = new G[points.len];
  for (u32 i = 0; i < points.len; i++)
    is >> points.ptr[i];
  return is;
}

template <typename G>
std::ostream &
operator<<(std::ostream &os, MsmPoints<G> &points)
{
  os << points.len << std::endl;
  for (u32 i = 0; i < points.len; i++)
    os << points.ptr[i] << std::endl;
  return os;
}

struct MsmScalars
{
  u32 len;
  alt_bn128_fr::Number *ptr;
};

std::istream &
operator>>(std::istream &is, MsmScalars &scalars)
{
  is >> scalars.len;
  std::cout << "Scalars len = " << scalars.len << std::endl;
  scalars.ptr = new alt_bn128_fr::Number[scalars.len];
  for (u32 i = 0; i < scalars.len; i++)
    is >> scalars.ptr[i];
  return is;
}

std::ostream &
operator<<(std::ostream &os, MsmScalars &scalars)
{
  os << scalars.len << std::endl;
  for (u32 i = 0; i < scalars.len; i++)
    os << scalars.ptr[i] << std::endl;
  return os;
}

template <typename G, typename GXYZZ>
float run_msm(MsmPoints<G> &points, MsmScalars &scalars, u32 len)
{

  cudaHostRegister((void *)scalars.ptr, scalars.len * sizeof(alt_bn128_fr::Number), cudaHostRegisterDefault);
  cudaHostRegister((void *)points.ptr, points.len * sizeof(G), cudaHostRegisterDefault);

  using Config = msm::MsmConfig<255, 16, 2, false>;
  u32 batch_size = 16;
  u32 batch_per_run = 1;
  u32 parts = 2;
  u32 stage_scalers = 2;
  u32 stage_points = 2;

  std::array<u32 *, Config::n_precompute> h_points;
  h_points[0] = (u32 *)points.ptr;
  for (u32 i = 1; i < Config::n_precompute; i++)
  {
    cudaHostAlloc(&h_points[i], len * sizeof(G), cudaHostAllocDefault);
  }

  std::vector<u32 *> scalers_batches;
  for (int i = 0; i < batch_size; i++)
  {
    scalers_batches.push_back((u32 *)scalars.ptr);
  }

  std::vector<GXYZZ> r(batch_size);

  std::vector<u32> cards;
  int card_count = 1;
  for (int i = 0; i < card_count; i++)
  {
    cards.push_back(i);
  }

  msm::MultiGPUMSM<Config, alt_bn128_fr::Number, GXYZZ, G> msm_solver(len, batch_per_run, parts, stage_scalers, stage_points, cards);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  msm::MSMPrecompute<Config, GXYZZ, G>::precompute(len, h_points, 4);
  msm_solver.set_points(h_points);

  msm_solver.alloc_gpu();
  cudaEvent_t start, stop;
  float elapsedTime = 0.0;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  msm_solver.msm(scalers_batches, r);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);

  cudaStreamDestroy(stream);

  for (int i = 0; i < batch_size; i++)
  {
    std::cout << r[i].to_affine() << std::endl;
  }

  std::cout << "Total cost time:" << elapsedTime << std::endl;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaHostUnregister((void *)scalars.ptr);
  cudaHostUnregister((void *)points.ptr);
  for (u32 i = 1; i < Config::n_precompute; i++)
  {
    cudaFreeHost(h_points[i]);
  }

  return elapsedTime;
}

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    std::cout << "usage: <prog> input_file_directory" << std::endl;
    return 2;
  }

  std::string dir = argv[1];

  std::ifstream params_if(dir + "/wit_params.txt");
  if (!params_if.is_open())
  {
    std::cout << "Cannot open params file" << std::endl;
    return 1;
  }
  u32 abc_commit_len, h_commit_len;
  params_if >> abc_commit_len;
  params_if >> h_commit_len;
  std::cout << "ABC commit len = " << abc_commit_len << std::endl;
  std::cout << "H commit len = " << h_commit_len << std::endl;

  std::ifstream a_query_if(dir + "/A_query.txt");
  MsmPoints<alt_bn128_g1::PointAffine> a_g1_points1;
  MsmPoints<alt_bn128_g1::PointAffine> a_g1_points2;
  std::cout << "Read A-query" << std::endl;
  a_query_if >> a_g1_points1;
  a_query_if >> a_g1_points2;

  std::ifstream b_query_if(dir + "/B_query.txt");
  MsmPoints<alt_bn128_g2::PointAffine> b_g2_points;
  MsmPoints<alt_bn128_g1::PointAffine> b_g1_points;
  std::cout << "Read B-query" << std::endl;
  b_query_if >> b_g2_points;
  b_query_if >> b_g1_points;

  std::ifstream c_query_if(dir + "/C_query.txt");
  MsmPoints<alt_bn128_g1::PointAffine> c_g1_points1;
  MsmPoints<alt_bn128_g1::PointAffine> c_g1_points2;
  std::cout << "Read C-query" << std::endl;
  c_query_if >> c_g1_points1;
  c_query_if >> c_g1_points2;

  std::ifstream h_query_if(dir + "/H_query.txt");
  MsmPoints<alt_bn128_g1::PointAffine> h_g1_points;
  std::cout << "Read H-query" << std::endl;
  h_query_if >> h_g1_points;

  std::ifstream k_query_if(dir + "/K_query.txt");
  MsmPoints<alt_bn128_g1::PointAffine> k_g1_points;
  std::cout << "Read K-query" << std::endl;
  k_query_if >> k_g1_points;

  std::ifstream abc_if(dir + "/ABCs.txt");
  MsmScalars abc_scalars;
  std::cout << "Read ABCs" << std::endl;
  abc_if >> abc_scalars;

  std::ifstream h_if(dir + "/Hs.txt");
  MsmScalars h_scalars;
  std::cout << "Read Hs" << std::endl;
  h_if >> h_scalars;

  float total_time = 0;

  std::cout << "Compute answer to A-query" << std::endl;
  total_time += run_msm<alt_bn128_g1::PointAffine, alt_bn128_g1::Point>(a_g1_points1, abc_scalars, abc_commit_len);
  total_time += run_msm<alt_bn128_g1::PointAffine, alt_bn128_g1::Point>(a_g1_points2, abc_scalars, abc_commit_len);

  std::cout << "Compute answer to B-query" << std::endl;
  total_time += run_msm<alt_bn128_g2::PointAffine, alt_bn128_g2::Point>(b_g2_points, abc_scalars, abc_commit_len);
  total_time += run_msm<alt_bn128_g1::PointAffine, alt_bn128_g1::Point>(b_g1_points, abc_scalars, abc_commit_len);

  std::cout << "Compute answer to C-query" << std::endl;
  total_time += run_msm<alt_bn128_g1::PointAffine, alt_bn128_g1::Point>(c_g1_points1, abc_scalars, abc_commit_len);
  total_time += run_msm<alt_bn128_g1::PointAffine, alt_bn128_g1::Point>(c_g1_points2, abc_scalars, abc_commit_len);

  std::cout << "Compute answer to H-query" << std::endl;
  total_time += run_msm<alt_bn128_g1::PointAffine, alt_bn128_g1::Point>(h_g1_points, h_scalars, h_commit_len);

  std::cout << "Compute answer to K-query" << std::endl;
  total_time += run_msm<alt_bn128_g1::PointAffine, alt_bn128_g1::Point>(k_g1_points, abc_scalars, abc_commit_len);

  std::cout << "Total cost time:" << total_time << std::endl;

  return 0;
}