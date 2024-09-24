#include "../src/mont.cuh"

#include <iostream>

using namespace mont256;

// 21888242871839275222246405745257275088696311157297823662689037894645226208583
__device__ Params params = {
    .m = BIG_INTEGER_CHUNKS8(0x30644e72, 0xe131a029, 0xb85045b6, 0x8181585d, 0x97816a91, 0x6871ca8d, 0x3c208c16, 0xd87cfd47),
    .r_mod = BIG_INTEGER_CHUNKS8(0xe0a77c1, 0x9a07df2f, 0x666ea36f, 0x7879462c, 0x0a78eb28, 0xf5c70b3d, 0xd35d438d, 0xc58f0d9d),
    .r2_mod = BIG_INTEGER_CHUNKS8(0x6d89f71, 0xcab8351f, 0x47ab1eff, 0x0a417ff6, 0xb5e71911, 0xd44501fb, 0xf32cfc5b, 0x538afa89),
    .m_prime = 3834012553};

__device__ const u32 a[8] = BIG_INTEGER_CHUNKS8(0x8e2f1b9, 0x74caa8b2, 0xa201f5ce, 0xdd06a772, 0x33525f1a, 0xc8794b1e, 0x460dd0e8, 0x3abe291e);

const u32 BATCH = 128;
const u32 THREADS = 768;
const u32 ITERS = 2;

__global__ void bench(Element *r)
{
  Env field(params);
  Element v = Element::load(a);
  for (u32 i = 0; i < BATCH; i++)
    v = field.mul(v, v);
  *r = v;
}

int main()
{
  float total_time = 0;

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  u32 grid_size = 64 * deviceProp.multiProcessorCount;

  for (u32 i = 0; i < ITERS; i++)
  {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    Element *r;
    cudaMalloc(&r, sizeof(Element));

    auto env = Env::host_new(params);

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
