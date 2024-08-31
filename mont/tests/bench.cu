#include "../src/mont.cuh"

#include <iostream>

using namespace mont256;

__device__ const auto params = Params{
    // 8749054268878081845992735117657085490803352298049958388812839746200388362933
    .m = BIG_INTEGER_CHUNKS8(0x1357ca0b, 0x1175f673, 0x618376f7, 0xbe5cb471, 0x29552c58, 0xfd07e66d, 0x5c09a3fc, 0x1951e2b5),
    .r_mod = BIG_INTEGER_CHUNKS8(0x48abd70, 0x1d027c24, 0x0c52f56b, 0x554ad640, 0xe6acbf7b, 0x26994c72, 0x5382ac32, 0xb6d77ccf),
    .r2_mod = BIG_INTEGER_CHUNKS8(0xaf6c7c1, 0x4cbccf2f, 0x2335d990, 0x8329a189, 0xd86803f4, 0x9da2f940, 0x61ee8dd3, 0x97f473f0),
    .m_prime = 2695922787};

__device__ const u32 a[8] = BIG_INTEGER_CHUNKS8(0x8e2f1b9, 0x74caa8b2, 0xa201f5ce, 0xdd06a772, 0x33525f1a, 0xc8794b1e, 0x460dd0e8, 0x3abe291e);

const u32 BATCH = 128;
const u32 THREADS = 256;
const u32 ITERS = 256;

__global__ void bench(u32 *r)
{
  Env field(params);
  auto na = Element::load(a);

  for (u32 i = 0; i < BATCH; i ++)
    na = field.mul(na, na);
  na.store(r);
}

int main()
{
  float total_time = 0;

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  u32 grid_size = 2 * deviceProp.multiProcessorCount;

  for (u32 i = 0; i < ITERS; i ++)
  {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    u32* r;
    cudaMalloc(&r, sizeof(Element));

    cudaEventRecord(start);
    bench<<<grid_size, THREADS>>>(r);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    total_time += elapsed_time;
  }

  std::cout << THREADS * BATCH * ITERS * grid_size / total_time * 1000 << std::endl;

  return 0;
}

