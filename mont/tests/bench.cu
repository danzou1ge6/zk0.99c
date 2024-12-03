#include "../src/bn256_fr.cuh"
#include "../src/field_tc.cuh"

#include <iostream>

using bn256_fr::Element;
using mont::u32;

const u32 BATCH = 128;
const u32 THREADS = 512;
const u32 ITERS = 2;

__global__ void bench(Element *r, const Element *a, const Element *b)
{
  Element v[4];
  for (u32 j = 0; j < 4; j ++)
    v[j] = b[j];
  for (u32 i = 0; i < BATCH; i++)
    for (u32 j = 0; j < 4; j ++)
      v[j] = v[j] * *a;
  for (u32 j = 0; j < 4; j ++)
    r[j] = v[j];
}

__global__ void bench_tc(Element *r, const Element *a, const Element *b)
{
  using namespace mont::tc256;
  u32 lane_id = threadIdx.x % 32;

  __shared__ u32 fas[8];
  FragmentA fa(fas);

  if (lane_id < 8)
    fas[lane_id] = a->n.limbs[lane_id];

  auto fb = FragmentB::load<0b1111>([b](u32 i, u32 j) { return b[i].n.limbs[j]; });

  FragmentW fr;
  for (u32 i = 0; i < BATCH; i++)
  {
    fr = mul<bn256_fr::Params>(fa, fb);
    fb = fr.transpose_to_b();
  }

  fr.store<0b1111>([r] (u32 i, u32 j, u32 w) { r[i].n.limbs[j] = w; });
}

template <typename F>
float time_it(u32 iters, F f)
{
  float total_time = 0;

  Element *r, *a, *b;
  cudaMalloc(&r, sizeof(Element) * 4);
  cudaMalloc(&a, sizeof(Element));
  cudaMalloc(&b, sizeof(Element) * 4);

  for (u32 i = 0; i < iters; i++)
  {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto ha = Element::host_random();
    Element hb[4];
    for (u32 i = 0; i < 4; i++)
      hb[i] = Element::host_random();
    cudaMemcpy(a, &ha, sizeof(Element), cudaMemcpyHostToDevice);
    cudaMemcpy(b, hb, sizeof(Element) * 4, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    f(r, a, b);
    cudaEventRecord(stop);

    auto err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
      std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
      std::exit(1);
    }

    Element hv[4] = {hb[0], hb[1], hb[2], hb[3]};
    for (u32 i = 0; i < BATCH; i ++)
      for (u32 j = 0; j < 4; j ++)
        hv[j] = hv[j] * ha;
    
    Element hr[4];
    cudaMemcpy(hr, r, sizeof(Element) * 4, cudaMemcpyDeviceToHost);

    for (u32 j = 0; j < 4; j ++)
      if (hr[j] != hv[j])
      {
        std::cout << "Computation error: " << ha.n << " ^ " << BATCH << " * " << hb[j].n << " = " << hv[j].n
          << ", but got " << hr[j] << std::endl;
        std::exit(1);
      }

    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    total_time += elapsed_time;
  }

  cudaFree(r);
  cudaFree(a);
  cudaFree(b);

  return total_time;
}

int main()
{

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  u32 grid_size = 8 * deviceProp.multiProcessorCount;

  float total_time = time_it(ITERS, [grid_size](Element *r, Element *a, Element *b)
                             { bench<<<grid_size, THREADS>>>(r, a, b); });
  std::cout << "CUDA Core  : " << THREADS * 4 * ITERS * BATCH * grid_size / total_time * 1000 << std::endl;

  float total_time_tc = time_it(ITERS, [grid_size](Element *r, Element *a, Element *b)
                                { bench_tc<<<grid_size, THREADS>>>(r, a, b); });
  std::cout << "Tensor Core: " << THREADS / 8 * ITERS * BATCH * grid_size / total_time_tc * 1000 << std::endl;
}
