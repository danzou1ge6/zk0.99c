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
  for (u32 j = 0; j < 4; j++)
    v[j] = b[j];
  for (u32 i = 0; i < BATCH; i++)
    for (u32 j = 0; j < 4; j++)
      v[j] = v[j] * *a;
  for (u32 j = 0; j < 4; j++)
    r[j] = v[j];
}

__global__ void bench_bn(Element *r, const Element *a, const Element *b)
{
  Element v[4];
  for (u32 j = 0; j < 4; j++)
    v[j] = b[j];
  for (u32 i = 0; i < BATCH; i++)
    for (u32 j = 0; j < 4; j++)
    {
      auto prod = v[j].n * a->n;
      memcpy(v[j].n.limbs, prod.limbs, 8 * sizeof(u32));
    }
  for (u32 j = 0; j < 4; j++)
    r[j] = v[j];
}

using mont::tc256::debug::Intermediates;

__global__ void bench_tc(Element *r, const Element *a, const Element *b, Intermediates *di, bool debug)
{
  using namespace mont::tc256;
  u32 lane_id = threadIdx.x % 32;
  u32 warp_id = threadIdx.x / 32;

  __shared__ Multiplier<bn256_fr::Params> mul;
  if (warp_id == 0)
    mul.load();

  __shared__ FragmentA fa[THREADS / 32];
  fa[warp_id].load(a->n.limbs);

  auto fb = FragmentB::load<0b1111>([b](u32 i, u32 j)
                                    { return b[i].n.limbs[j]; });

  FragmentW fr;
  if (debug)
  {
    for (u32 i = 0; i < BATCH; i++)
    {
      if (warp_id == 0 && blockIdx.x == 0)
        fr = mul.template execute<true>(fa[warp_id], fb, di);
      else
        fr = mul(fa[warp_id], fb);
      fb = fr.transpose_to_b();
    }
  }
  else
  {
    for (u32 i = 0; i < BATCH; i++)
    {
      fr = mul(fa[warp_id], fb);
      fb = fr.transpose_to_b();
    }
  }

  if (warp_id == 0 && blockIdx.x == 0)
    fr.store<0b1111>([r](u32 i, u32 j, u32 w)
                     { r[i].n.limbs[j] = w; });
}

__global__ void bench_bn_tc(Element *r, const Element *a, const Element *b, Intermediates *di, bool debug)
{
  using namespace mont::tc256;
  u32 lane_id = threadIdx.x % 32;
  u32 warp_id = threadIdx.x / 32;

  __shared__ FragmentA fa[THREADS / 32];
  fa[warp_id].load(a->n.limbs);

  auto fb = FragmentB::load<0b1111>([b](u32 i, u32 j)
                                    { return b[i].n.limbs[j]; });

  FragmentW fr;
  if (debug)
  {
    for (u32 i = 0; i < BATCH; i++)
    {
      if (warp_id == 0 && blockIdx.x == 0)
        fr = number_multiplication<true>(fa[warp_id], fb, di);
      else
        fr = number_multiplication<false>(fa[warp_id], fb);
      fb = fr.transpose_to_b();
    }
  }
  else
  {
    for (u32 i = 0; i < BATCH; i++)
    {
      fr = number_multiplication<false>(fa[warp_id], fb);
      fb = fr.transpose_to_b();
    }
  }

  if (warp_id == 0 && blockIdx.x == 0)
    fr.store<0b1111>([r](u32 i, u32 j, u32 w)
                     { r[i].n.limbs[j] = w; });
}

template <typename F, typename F1>
float time_it(u32 iters, F f, F1 op, bool print_intermediates)
{
  float total_time = 0;

  Element *r, *a, *b;
  cudaMalloc(&r, sizeof(Element) * 4);
  cudaMalloc(&a, sizeof(Element));
  cudaMalloc(&b, sizeof(Element) * 4);

  auto intermediates = Intermediates::new_device();
  Intermediates *d_intermediates;
  cudaMalloc(&d_intermediates, sizeof(Intermediates));
  cudaMemcpy(d_intermediates, &intermediates, sizeof(Intermediates), cudaMemcpyHostToDevice);

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
    f(r, a, b, d_intermediates);
    cudaEventRecord(stop);

    auto err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
      std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
      std::exit(1);
    }

    Element hv[4] = {hb[0], hb[1], hb[2], hb[3]};
    for (u32 i = 0; i < BATCH; i++)
      for (u32 j = 0; j < 4; j++)
        hv[j] = op(hv[j], ha);

    Element hr[4];
    cudaMemcpy(hr, r, sizeof(Element) * 4, cudaMemcpyDeviceToHost);

    for (u32 j = 0; j < 4; j++)
      if (hr[j] != hv[j])
      {
        std::cout << "Computation error at iteration " << std::dec << i << " : "
                  << ha.n << " ^ " << std::dec << BATCH << " * " << hb[j].n << " = " << hv[j].n
                  << ", but got " << hr[j] << std::endl;
        if (print_intermediates)
          std::cout << "Intermediates:" << std::endl
                    << intermediates.to_host();
        // std::exit(1);
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

int main(int argc, char *argv[])
{
  bool debug;
  if (argc == 2 && strcmp(argv[1], "debug") == 0)
    debug = true;
  else
    debug = false;

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  u32 grid_size = 8 * deviceProp.multiProcessorCount;

  std::cout << "Montgomery Multiplication" << std::endl;

  auto mmul = [](Element &x, Element &y) { return x * y;};

  float total_time = time_it(ITERS, [grid_size](Element *r, Element *a, Element *b, Intermediates *di)
                             { bench<<<grid_size, THREADS>>>(r, a, b); }, mmul, false);
  std::cout << "CUDA Core  : " << THREADS * 4 * ITERS * BATCH * grid_size / total_time * 1000 << std::endl;

  float total_time_tc = time_it(ITERS, [grid_size, debug](Element *r, Element *a, Element *b, Intermediates *di)
                                { bench_tc<<<grid_size, THREADS>>>(r, a, b, di, debug); }, mmul, debug);
  std::cout << "Tensor Core: " << THREADS / 8 * ITERS * BATCH * grid_size / total_time_tc * 1000 << std::endl;

  std::cout << "Big Number Multiplication" << std::endl;

  auto bnmul = [](Element &x, Element &y)
  {
    auto prod = x.n * y.n;
    Element r;
    memcpy(r.n.limbs, prod.limbs, 8 * sizeof(u32));
    return r;
  };

  float total_time_bn = time_it(ITERS, [grid_size](Element *r, Element *a, Element *b, Intermediates *di)
                                       { bench_bn<<<grid_size, THREADS>>>(r, a, b); }, bnmul, false);
  std::cout << "CUDA Core  : " << THREADS * 4 * ITERS * BATCH * grid_size / total_time_bn * 1000 << std::endl;

  float total_time_bn_tc = time_it(ITERS, [grid_size, debug](Element *r, Element *a, Element *b, Intermediates *di)
                                         { bench_bn_tc<<<grid_size, THREADS>>>(r, a, b, di, debug); }, bnmul, debug);
  std::cout << "Tensor Core: " << THREADS / 8 * ITERS * BATCH * grid_size / total_time_bn_tc * 1000 << std::endl;
}
