#include "../../../NTT/src/NTT.cuh"
#include "./ntt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// // 3221225473
// const auto params = mont256::Params {
//   .m = BIG_INTEGER_CHUNKS8(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xc0000001),
//   .r_mod = BIG_INTEGER_CHUNKS8(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x9fc05273),
//   .r2_mod = BIG_INTEGER_CHUNKS8(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x9c229677),
//   .m_prime = 3221225471
// };

// 28948022309329048855892746252171976963363056481941560715954676764349967630337
const auto params = mont256::Params {
  .m = BIG_INTEGER_CHUNKS8(0x40000000, 0x00000000, 0x00000000, 0x00000000, 0x224698fc, 0x094cf91b, 0x992d30ed, 0x00000001),
  .r_mod = BIG_INTEGER_CHUNKS8(0x3fffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x992c350b, 0xe41914ad, 0x34786d38, 0xfffffffd),
  .r2_mod = BIG_INTEGER_CHUNKS8(0x96d41af, 0x7b9cb714, 0x7797a99b, 0xc3c95d18, 0xd7d30dbd, 0x8b0de0e7, 0x8c78ecb3, 0x0000000f),
  .m_prime = 4294967295
};

void cuda_ntt(unsigned int *data, const unsigned int *omega, int log_n) {
    NTT::self_sort_in_place_ntt<NUM_OF_UINT> SSIP(params, omega, log_n, false);
    SSIP.ntt(data);
}

#ifdef __cplusplus
}
#endif