#include "../src/bn256_fr.cuh"
#include "../src/field_tc.cuh"

#include <iostream>

using bn256_fr::Element;

__global__ void kernel(
    const Element *x,
    const Element *y,
    Element *z,
    mont::tc256::debug::Intermediates *i)
{
  using namespace mont::tc256;
  using mont::u32;

  __shared__ Multiplier<bn256_fr::Params> mul;
  mul.load();

  __shared__ FragmentA fx; // only one warp here
  fx.load(x->n.limbs);

  auto fy = FragmentB::load<0b1111>([y](u32 i, u32 j)
                                    { return y[i].n.limbs[j]; });
  auto fz = mul.template execute<true>(fx, fy, i);
  fz.store<0b1111>([z](u32 i, u32 j, u32 w)
                   { z[i].n.limbs[j] = w; });
}

int main()
{
  Element x = mont::Number<8>(BIG_INTEGER_CHUNKS8(0x2fa3e185, 0x1fa39e5a, 0x168b3cbd, 0x0d33b74a, 0x086f10d9, 0x032c7039, 0x6195cc82, 0x0f9e13e6));
  Element y[4] = {
      mont::Number<8>(BIG_INTEGER_CHUNKS8(0xd9fed2, 0x2c958bd3, 0x14383fb8, 0x55e308c6, 0x024f5623, 0x39d0d22a, 0x4aac4454, 0x3074fa73)),
      mont::Number<8>(BIG_INTEGER_CHUNKS8(0xd9fed2, 0x2c958bd3, 0x14383fb8, 0x55e308c6, 0x024f5623, 0x39d0d22a, 0x4aac4454, 0x3074fa73)),
      mont::Number<8>(BIG_INTEGER_CHUNKS8(0xd9fed2, 0x2c958bd3, 0x14383fb8, 0x55e308c6, 0x024f5623, 0x39d0d22a, 0x4aac4454, 0x3074fa73)),
      mont::Number<8>(BIG_INTEGER_CHUNKS8(0xd9fed2, 0x2c958bd3, 0x14383fb8, 0x55e308c6, 0x024f5623, 0x39d0d22a, 0x4aac4454, 0x3074fa73)),
  };

  // Element x = mont::Number<8>(BIG_INTEGER_CHUNKS8(0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101));
  // Element y[4] = {
  //     mont::Number<8>(BIG_INTEGER_CHUNKS8(0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101)),
  //     mont::Number<8>(BIG_INTEGER_CHUNKS8(0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101)),
  //     mont::Number<8>(BIG_INTEGER_CHUNKS8(0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101)),
  //     mont::Number<8>(BIG_INTEGER_CHUNKS8(0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101)),
  // };

  Element z[4];
  auto i = mont::tc256::debug::Intermediates::new_device();

  Element *dx, *dy, *dz;
  mont::tc256::debug::Intermediates *di;
  cudaMalloc(&dx, sizeof(Element));
  cudaMalloc(&dy, sizeof(Element) * 4);
  cudaMalloc(&dz, sizeof(Element) * 4);
  cudaMalloc(&di, sizeof(mont::tc256::debug::Intermediates));
  cudaMemcpy(dx, &x, sizeof(Element), cudaMemcpyHostToDevice);
  cudaMemcpy(dy, y, sizeof(Element) * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(di, &i, sizeof(mont::tc256::debug::Intermediates), cudaMemcpyHostToDevice);

  for (int i = 0; i < 256; i ++)
    kernel<<<1, 32>>>(dx, dy, dz, di);
  auto err = cudaStreamSynchronize(0);
  if (err != cudaSuccess)
  {
    std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    return 1;
  }

  cudaMemcpy(&z, dz, sizeof(Element) * 4, cudaMemcpyDeviceToHost);

  std::cout << "Correct answer = " << (x * y[0]).n << std::endl;
  std::cout << "Got            = " << z[0].n << std::endl;
  std::cout << i.to_host();

  return 0;
}
