#include "../src/bn256_fr.cuh"
#include "../src/field_tc.cuh"

#include <iostream>

using bn256_fr::Element;
using mont::u32;

const u32 BATCH = 3;
const u32 THREADS = 256;
const u32 ITERS = 2;

__global__ void bench(Element *r, const Element *a, const Element *b)
{
  auto v0 = *a;
  auto v1 = *a;
  auto v2 = *a;
  auto v3 = *a;
  for (u32 i = 0; i < BATCH; i++)
  {
    v0 = v0 * b[0];
    v1 = v1 * b[1];
    v2 = v2 * b[2];
    v3 = v3 * b[3];
  }
  r[0] = v0;
  r[1] = v1;
  r[2] = v2;
  r[3] = v3;
}

__global__ void bench_tc(Element *r, const Element *a, const Element *b)
{
  using namespace mont::tc256;
  u32 lane_id = threadIdx.x % 32;

  __shared__ FragmentA fa;
  if (lane_id < 8)
    fa.a[lane_id] = a->n.limbs[lane_id];

  mont::Reference rb[4] = {
      b[0].n.limbs.to_ref(),
      b[1].n.limbs.to_ref(),
      b[2].n.limbs.to_ref(),
      b[3].n.limbs.to_ref(),
  };

  mont::Reference rr[4] = {
      r[0].n.limbs.to_ref(),
      r[1].n.limbs.to_ref(),
      r[2].n.limbs.to_ref(),
      r[3].n.limbs.to_ref(),
  };

  auto fb = FragmentB::load<0b1111>(rb);

  FragmentW fr;
  fr = mul<bn256_fr::Params>(fa, fb);

  fr.store<0b1111>(rr);
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

    auto err = cudaStreamSynchronize(0);
    if (err != cudaSuccess)
    {
      std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
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

  // float total_time = time_it(ITERS, [grid_size](Element *r, Element *a, Element *b)
  //                            { bench<<<grid_size, THREADS>>>(r, a, b); });
  // std::cout << "CUDA Core  : " << THREADS * 4 * ITERS * BATCH * grid_size / total_time * 1000 << std::endl;

  float total_time_tc = time_it(ITERS, [grid_size](Element *r, Element *a, Element *b)
                                { bench_tc<<<grid_size, THREADS>>>(r, a, b); });
  std::cout << "Tensor Core: " << THREADS / 8 * ITERS * BATCH * grid_size / total_time_tc * 1000 << std::endl;
}
