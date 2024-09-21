#pragma once
#ifdef __cplusplus
extern "C" {
#endif

enum FIELD {
    PASTA_CURVES_FIELDS_FP,
    HALO2CURVES_BN256_FR
};

void cuda_ntt(unsigned int *data, const unsigned int *omega, unsigned int log_n, FIELD field);

#ifdef __cplusplus
}
#endif