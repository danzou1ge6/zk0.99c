#pragma once
#ifdef __cplusplus
extern "C" {
#endif

void cuda_ntt(unsigned int *data, const unsigned int *omega, int log_n);

#ifdef __cplusplus
}
#endif