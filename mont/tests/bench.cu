#include "../src/bn256_fr.cuh"
#include "../src/field_tc.cuh"

#include <iostream>

using bn256_fr::Element;
using mont::u32;

const u32 BATCH = 128;
const u32 THREADS = 256;
const u32 ITERS = 2;

__global__ void bench(Element *r, const Element *a, const Element *b)
{
  r[0] = *a * b[0];
  r[1] = *a * b[1];
  r[2] = *a * b[2];
  r[3] = *a * b[3];
}

__global__ void bench_tc(Element *r, const Element *a, const Element *b)
{
  // __shared__ Element v[THREADS / 32];
  // u32 lane_id = threadIdx.x % 32;
  // u32 warp_id = threadIdx.x / 32;
  // if (lane_id < 8)
  //   v[warp_id].n.limbs[lane_id] = a->n.limbs[lane_id];

  // for (u32 i = 0; i < BATCH; i++)
  // {
  //   const mont::Reference st_y[1] = {
  //       v[warp_id].n.limbs.to_ref()};
  //   mont::Reference st_z[1] = {
  //       v[warp_id].n.limbs.to_ref()};
  //   mont::tc256::mul<1, false, bn256_fr::Params>(st_z, mont::Reference((mont::u32 *)a), st_y);
  // }

  // if (lane_id < 8)
  //   r->n.limbs[lane_id] = v[warp_id].n.limbs[lane_id];
  const mont::Reference st_y[4] = {
    mont::Reference((u32*)b),
    mont::Reference((u32*)(b + 1)),
    mont::Reference((u32*)(b + 2)),
    mont::Reference((u32*)(b + 3))
  };
  mont::Reference st_z[4] = {
    mont::Reference((u32*)r),
    mont::Reference((u32*)(r + 1)),
    mont::Reference((u32*)(r + 2)),
    mont::Reference((u32*)(r + 3)),
  };
  mont::tc256::mul<4, false, bn256_fr::Params>(st_z, mont::Reference((u32*)a), st_y);
}

template <typename F>
float time_it(u32 iters, F f)
{
  float total_time = 0;

  for (u32 i = 0; i < iters; i++)
  {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    Element *r, *a, *b;
    cudaMalloc(&r, sizeof(Element) * 4);
    cudaMalloc(&a, sizeof(Element));
    cudaMalloc(&b, sizeof(Element) * 4);

    auto ha = Element::host_random();
    Element hb[4];
    for (u32 i = 0; i < 4; i ++)
      hb[i] = Element::host_random();
    cudaMemcpy(a, &ha, sizeof(Element), cudaMemcpyHostToDevice);
    cudaMemcpy(b, hb, sizeof(Element) * 4, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    f(r, a, b);
    cudaEventRecord(stop);

    auto err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
      std::exit(1);
    }

    cudaFree(r);
    cudaFree(a);
    cudaFree(b);

    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    total_time += elapsed_time;
  }

  return total_time;
}

int main()
{

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  u32 grid_size = 32 * deviceProp.multiProcessorCount;

  float total_time = time_it(ITERS, [grid_size](Element *r, Element *a, Element *b)
                             { bench<<<grid_size, THREADS>>>(r, a, b); });
  std::cout << "CUDA Core  : " << THREADS * 4 * ITERS * grid_size / total_time * 1000 << std::endl;

  total_time = time_it(ITERS, [grid_size](Element *r, Element *a, Element *b)
                             { bench_tc<<<grid_size, THREADS>>>(r, a, b); });
  std::cout << "Tensor Core: " << THREADS / 8 * ITERS * grid_size / total_time * 1000 << std::endl;
}
