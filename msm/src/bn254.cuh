#ifndef BN254_H
#define BN254_H

#include "../../mont/src/mont.cuh"
#include "./curve.cuh"

namespace bn254
{

  using namespace curve256;
  using mont256::u32;

  namespace fq
  {
    // 21888242871839275222246405745257275088696311157297823662689037894645226208583
    __device__ mont256::Params PARAMS = {
        .m = BIG_INTEGER_CHUNKS8(0x30644e72, 0xe131a029, 0xb85045b6, 0x8181585d, 0x97816a91, 0x6871ca8d, 0x3c208c16, 0xd87cfd47),
        .r_mod = BIG_INTEGER_CHUNKS8(0xe0a77c1, 0x9a07df2f, 0x666ea36f, 0x7879462c, 0x0a78eb28, 0xf5c70b3d, 0xd35d438d, 0xc58f0d9d),
        .r2_mod = BIG_INTEGER_CHUNKS8(0x6d89f71, 0xcab8351f, 0x47ab1eff, 0x0a417ff6, 0xb5e71911, 0xd44501fb, 0xf32cfc5b, 0x538afa89),
        .m_prime = 3834012553
    };
  }

  // 3 in Montgomery representation
  __device__ const u32 G1B[8] = BIG_INTEGER_CHUNKS8(
      0x2a1f6744, 0xce179d8e, 0x334bea4e, 0x696bd284, 0x1f6ac17a, 0xe15521b9, 0x7a17caa9, 0x50ad28d7);
  // 9 in Montgomery representation
  __device__ const u32 G1B3[8] = BIG_INTEGER_CHUNKS8(
      0x1d9598e8, 0xa7e39857, 0x2943337e, 0x3940c6d1, 0x2f3d6f4d, 0xd31bd011, 0xf60647ce, 0x410d7ff7);

  __device__ __forceinline__ Curve new_bn254()
  {
    auto field = mont256::Env(fq::PARAMS);
    auto a = field.zero();
    auto b = Element::load(G1B);
    auto b3 = Element::load(G1B3);
    return Curve(field, a, b, b3);
  }
}

#endif