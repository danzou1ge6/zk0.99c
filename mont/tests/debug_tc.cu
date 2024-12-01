#include "../src/bn256_fr.cuh"
#include "../src/field_tc.cuh"

#include <iostream>

using bn256_fr::Element;

__global__ void kernel(const Element *x, const Element *y, Element *z, mont::tc256::debug::Intermediates *i)
{
  const mont::Reference st_y[4] = {
      mont::Reference((mont::u32 *)y),
      mont::Reference((mont::u32 *)(y + 1)),
      mont::Reference((mont::u32 *)(y + 2)),
      mont::Reference((mont::u32 *)(y + 3)),
  };
  mont::Reference st_z[4] = {
      mont::Reference((mont::u32 *)z),
      mont::Reference((mont::u32 *)(z + 1)),
      mont::Reference((mont::u32 *)(z + 2)),
      mont::Reference((mont::u32 *)(z + 3)),
  };
  using namespace mont::tc256;

  FragmentA fx(x->n.limbs.to_ref());
  auto fy = FragmentB::load<0b1111>(st_y);
  auto fz = mul<bn256_fr::Params, true>(fx, fy, i);
  fz.store<0b1111>(st_z);
  auto test = fz.transpose_to_b();
  debug::store_b_matrix(test.b0, test.b1, i->ub);
}

int main()
{
  Element x = mont::Number<8>(BIG_INTEGER_CHUNKS8(0x20b962c2, 0xed033696, 0xa5384118, 0x06c215e1, 0xbb31704c, 0x6a92ae0e, 0x31f7b2f9, 0x1bf8bb19));
  Element y[4] = {
      mont::Number<8>(BIG_INTEGER_CHUNKS8(0x1977f26e, 0x7eb3df21, 0x25ffd6e7, 0x558f2112, 0xc1358e32, 0x98c44536, 0x5528af35, 0xe0a0b93b)),
      mont::Number<8>(BIG_INTEGER_CHUNKS8(0x1977f26e, 0x7eb3df21, 0x25ffd6e7, 0x558f2112, 0xc1358e32, 0x98c44536, 0x5528af35, 0xe0a0b93b)),
      mont::Number<8>(BIG_INTEGER_CHUNKS8(0x1977f26e, 0x7eb3df21, 0x25ffd6e7, 0x558f2112, 0xc1358e32, 0x98c44536, 0x5528af35, 0xe0a0b93b)),
      mont::Number<8>(BIG_INTEGER_CHUNKS8(0x1977f26e, 0x7eb3df21, 0x25ffd6e7, 0x558f2112, 0xc1358e32, 0x98c44536, 0x5528af35, 0xe0a0b93b)),
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

  std::cout << z[0].n << std::endl;
  std::cout << i.to_host();

  return 0;
}
