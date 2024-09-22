#include "../../../ntt/src/self_sort_in_place_ntt.cuh"
#include "./ntt_c_api.h"
#include <cuda_runtime.h>

// pasta_fp
// 28948022309329048855892746252171976963363056481941560715954676764349967630337
const auto params_pasta_fp = mont256::Params {
  .m = BIG_INTEGER_CHUNKS8(0x40000000, 0x00000000, 0x00000000, 0x00000000, 0x224698fc, 0x094cf91b, 0x992d30ed, 0x00000001),
  .r_mod = BIG_INTEGER_CHUNKS8(0x3fffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x992c350b, 0xe41914ad, 0x34786d38, 0xfffffffd),
  .r2_mod = BIG_INTEGER_CHUNKS8(0x96d41af, 0x7b9cb714, 0x7797a99b, 0xc3c95d18, 0xd7d30dbd, 0x8b0de0e7, 0x8c78ecb3, 0x0000000f),
  .m_prime = 4294967295
};

// bn256_fr
// 21888242871839275222246405745257275088548364400416034343698204186575808495617
const auto params_bn256_fr = mont256::Params {
  .m = BIG_INTEGER_CHUNKS8(0x30644e72, 0xe131a029, 0xb85045b6, 0x8181585d, 0x2833e848, 0x79b97091, 0x43e1f593, 0xf0000001),
  .r_mod = BIG_INTEGER_CHUNKS8(0xe0a77c1, 0x9a07df2f, 0x666ea36f, 0x7879462e, 0x36fc7695, 0x9f60cd29, 0xac96341c, 0x4ffffffb),
  .r2_mod = BIG_INTEGER_CHUNKS8(0x216d0b1, 0x7f4e44a5, 0x8c49833d, 0x53bb8085, 0x53fe3ab1, 0xe35c59e3, 0x1bb8e645, 0xae216da7),
  .m_prime = 4026531839
};

#ifdef __cplusplus
extern "C" {
#endif

bool cuda_ntt(unsigned int *data, const unsigned int *omega, unsigned int log_n, FIELD FIELD) {
    cudaError_t err;
    if (FIELD == FIELD::PASTA_CURVES_FIELDS_FP) {
      ntt::self_sort_in_place_ntt<8> SSIP(params_pasta_fp, omega, log_n, false);
      err = SSIP.ntt(data);
    } else if (FIELD == FIELD::HALO2CURVES_BN256_FR) {
      ntt::self_sort_in_place_ntt<8> SSIP(params_bn256_fr, omega, log_n, false);
      err = SSIP.ntt(data);
    } else {
      return false;
    }
    return err == cudaSuccess;
}

#ifdef __cplusplus
}
#endif