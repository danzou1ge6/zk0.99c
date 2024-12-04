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

  __shared__ Multiplier<bn256_fr::Params, true> mul;
  mul.load();

  __shared__ FragmentA fx; // only one warp here
  fx.load(x->n.limbs);

  auto fy = FragmentB::load<0b1111>([y](u32 i, u32 j)
                                    { return y[i].n.limbs[j]; });
  auto fz = mul(fx, fy, i);
  fz.store<0b1111>([z](u32 i, u32 j, u32 w)
                   { z[i].n.limbs[j] = w; });
}

int main()
{
  Element x = mont::Number<8>(BIG_INTEGER_CHUNKS8(0xfa33c07, 0x55497a85, 0x58972cab, 0x3f42f3af, 0x0746f5fe, 0x6b3a3cd2, 0x2b2542f9, 0x6e9a0ff8));
  Element y[4] = {
      mont::Number<8>(BIG_INTEGER_CHUNKS8(0x2f8b39bb, 0x0cb6ab08, 0x5bd50c97, 0x36d22fb9, 0x77f0e7da, 0x06fa4f90, 0x256c3fb2, 0x736cdb07)),
      mont::Number<8>(BIG_INTEGER_CHUNKS8(0x2f8b39bb, 0x0cb6ab08, 0x5bd50c97, 0x36d22fb9, 0x77f0e7da, 0x06fa4f90, 0x256c3fb2, 0x736cdb07)),
      mont::Number<8>(BIG_INTEGER_CHUNKS8(0x2f8b39bb, 0x0cb6ab08, 0x5bd50c97, 0x36d22fb9, 0x77f0e7da, 0x06fa4f90, 0x256c3fb2, 0x736cdb07)),
      mont::Number<8>(BIG_INTEGER_CHUNKS8(0x2f8b39bb, 0x0cb6ab08, 0x5bd50c97, 0x36d22fb9, 0x77f0e7da, 0x06fa4f90, 0x256c3fb2, 0x736cdb07)),
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
