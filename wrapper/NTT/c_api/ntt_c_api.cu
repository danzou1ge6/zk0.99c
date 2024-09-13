#include "../../../NTT/src/NTT.cuh"
#include "./ntt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// 3221225473
const auto params = mont256::Params {
  .m = BIG_INTEGER_CHUNKS8(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xc0000001),
  .r_mod = BIG_INTEGER_CHUNKS8(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x9fc05273),
  .r2_mod = BIG_INTEGER_CHUNKS8(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x9c229677),
  .m_prime = 3221225471
};

void cuda_ntt(unsigned int *data, const unsigned int *omega, int log_n) {
    NTT::self_sort_in_place_ntt<NUM_OF_UINT> SSIP(params, omega, log_n, true);
    SSIP.ntt(data);
}

#ifdef __cplusplus
}
#endif