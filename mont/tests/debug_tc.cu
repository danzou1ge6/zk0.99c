#include "../src/bn256_fr.cuh"
#include "../src/field_tc.cuh"

#include <iostream>

using bn256_fr::Element;

__global__ void kernel(const Element *x, const Element *y, Element *z, mont::tc256::debug::Intermediates *i)
{
  // const mont::Reference st_y[4] = {
  //     mont::Reference((mont::u32 *)y),
  //     mont::Reference((mont::u32 *)(y + 1)),
  //     mont::Reference((mont::u32 *)(y + 2)),
  //     mont::Reference((mont::u32 *)(y + 3)),
  // };
  // mont::Reference st_z[4] = {
  //     mont::Reference((mont::u32 *)z),
  //     mont::Reference((mont::u32 *)(z + 1)),
  //     mont::Reference((mont::u32 *)(z + 2)),
  //     mont::Reference((mont::u32 *)(z + 3)),
  // };
  // const mont::Reference st_x = mont::Reference((mont::u32 *)x);
  // mont::tc256::mul<4, true, bn256_fr::Params>(st_z, st_x, st_y, i);
  using namespace mont::tc256;
  using mont::u32;

  u32 a0, a1, a2, a3;
  debug::polulate_a_matrix(a0, a1, a2, a3, [](u32 i, u32 j) { return j; });
  u32 b0, b1;
  debug::polulate_b_matrix(b0, b1, [](u32 i, u32 j) { return (i == j) ? 1 : 0; });
  u32 d0, d1, d2, d3;
  mma_m16n8k32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);
  debug::store_a_matrix(a0, a1, a2, a3, i->xa0);
  debug::store_b_matrix(b0, b1, i->yb);
  debug::store_d_matrix(d0, d1, d2, d3, i->sd0);
}

int main()
{
  Element x = mont::Number<8>(BIG_INTEGER_CHUNKS8(1, 1, 1, 1, 1, 1, 1, 1));
  Element y[4] = {
      mont::Number<8>(BIG_INTEGER_CHUNKS8(1, 1, 1, 1, 1, 1, 1, 1)),
      mont::Number<8>(BIG_INTEGER_CHUNKS8(1, 1, 1, 1, 1, 1, 1, 1)),
      mont::Number<8>(BIG_INTEGER_CHUNKS8(1, 1, 1, 1, 1, 1, 1, 1)),
      mont::Number<8>(BIG_INTEGER_CHUNKS8(1, 1, 1, 1, 1, 1, 1, 1)),
  };
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

  std::cout << z[0] << std::endl;
  std::cout << i.to_host();

  return 0;
}
