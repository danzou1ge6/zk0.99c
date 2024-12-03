#include "../src/bn256_fr.cuh"
#include "../src/field_tc.cuh"

#include <iostream>

using bn256_fr::Element;

__global__ void kernel(const Element *x, const Element *y, Element *z, mont::tc256::debug::Intermediates *i)
{
  using namespace mont::tc256;
  using mont::u32;

  __shared__ Multiplier<bn256_fr::Params, true> mul;
  mul.load();

  __shared__ FragmentA fx;  // only one warp here
  fx.load(x->n.limbs);

  auto fy = FragmentB::load<0b1111>([y](u32 i, u32 j) { return y[i].n.limbs[j]; });
  auto fz = mul(fx, fy, i);
  fz.store<0b1111>([z](u32 i, u32 j, u32 w) { z[i].n.limbs[j] = w; });
}

int main()
{
  Element x = mont::Number<8>(BIG_INTEGER_CHUNKS8(0x06074b4b, 0x1df79173, 0x3c133ef9, 0x1819d4bc, 0xd33fac94, 0xe36715f1, 0x7779c165, 0xd12e658d));
  Element y[4] = {
      mont::Number<8>(BIG_INTEGER_CHUNKS8(0x1457b41b, 0xc2455063, 0x1b0a7958, 0xa4803a05, 0x755211e3, 0xa13bbbd6, 0x5be452ae, 0x7e785885)),
      mont::Number<8>(BIG_INTEGER_CHUNKS8(0x1457b41b, 0xc2455063, 0x1b0a7958, 0xa4803a05, 0x755211e3, 0xa13bbbd6, 0x5be452ae, 0x7e785885)),
      mont::Number<8>(BIG_INTEGER_CHUNKS8(0x1457b41b, 0xc2455063, 0x1b0a7958, 0xa4803a05, 0x755211e3, 0xa13bbbd6, 0x5be452ae, 0x7e785885)),
      mont::Number<8>(BIG_INTEGER_CHUNKS8(0x1457b41b, 0xc2455063, 0x1b0a7958, 0xa4803a05, 0x755211e3, 0xa13bbbd6, 0x5be452ae, 0x7e785885)),
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
