#include "../src/bn254_scalar.cuh"

#include <iostream>

using bn254_scalar::Element;
using mont::u32;

const u32 BATCH = 128;
const u32 THREADS = 512;
const u32 ITERS = 2;

__global__ void bench(Element *r, const Element *a)
{
  Element v = *a;
  for (u32 i = 0; i < BATCH; i++)
  {
    v = v * v;
  }
  *r = v;
}

int main()
{
  float total_time = 0;

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  u32 grid_size = 32 * deviceProp.multiProcessorCount;

  for (u32 i = 0; i < ITERS; i++)
  {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    Element *r, *a;
    cudaMalloc(&r, sizeof(Element));
    cudaMalloc(&a, sizeof(Element));

    auto ha = Element::host_random();
    cudaMemcpy(a, &ha, sizeof(Element), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    bench<<<grid_size, THREADS>>>(r, a);
    cudaEventRecord(stop);

    auto err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
      return 1;
    }

    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    total_time += elapsed_time;
  }

  std::cout << THREADS * BATCH * ITERS * grid_size / total_time * 1000 << std::endl;

  return 0;
}
