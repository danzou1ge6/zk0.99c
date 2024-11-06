#include <iostream>
#include <ctime>

#include "../src/bn256_fr.cuh"
using bn256_fr::Element;

using Number = mont::Number<8>;
using Number2 = mont::Number<16>;
using mont::u32;
using mont::u64;
using mont::usize;

void print_progress(float progress) {
    constexpr int barWidth = 50;

    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << progress * 100 << " %\r";
    std::cout.flush();
}

__global__ void mont_mul(Element *r, const Element *a, const Element *b, usize len)
{
  usize tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < len)
    r[tid] = a[tid] * b[tid];
}

usize pow2(usize n)
{
  return n == 0 ? 1 : 2 * pow2(n - 1);
}

int main(int argc, char *argv[])
{
  if (argc != 2) {
    std::cout << "usage: <prog> k" << std::endl;
    return 2;
  }
  usize k = std::atoi(argv[1]);

  usize grid_size = 128;
  usize block_size = 512;
  usize len = grid_size * block_size;
  usize iters = pow2(k);

  Element *dr, *da, *db;
  cudaMalloc(&dr, sizeof(Element) * len);
  cudaMalloc(&da, sizeof(Element) * len);
  cudaMalloc(&db, sizeof(Element) * len);

  Element *expected = new Element[len];
  Element *a = new Element[len];
  Element *b = new Element[len];
  Element *r = new Element[len];

  for (int it = 0; it < iters; it++)
  {
    std::srand(std::time(nullptr));
    for (int i = 0; i < len; i++)
    {
      a[i] = Element::host_random();
      b[i] = Element::host_random();
      expected[i] = a[i] * b[i];
    }

    cudaMemcpy(da, a, sizeof(Element) * len, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, sizeof(Element) * len, cudaMemcpyHostToDevice);

    mont_mul<<<grid_size, block_size>>>(dr, da, db, len);

    cudaMemcpy(r, dr, sizeof(Element) * len, cudaMemcpyDeviceToHost);

    for (int i = 0; i < len; i++)
    {
      if (r[i] != expected[i])
      {
        std::cout << "Expected " << a[i] << " * " << b[i] << " = " << expected[i] << ", got" << r[i] << std::endl;
        return 1;
      }
    }

    print_progress((float)it / iters);
  }
  return 0;
}