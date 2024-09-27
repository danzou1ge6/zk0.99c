#pragma once

enum FIELD {
    PASTA_CURVES_FIELDS_FP,
    HALO2CURVES_BN256_FR
};

bool cuda_ntt(unsigned int *data, const unsigned int *omega, unsigned int log_n, FIELD field, bool inverse, bool process, const unsigned int * inv_n, const unsigned int * zeta);

bool cuda_coeff_to_extended(unsigned int *data, const unsigned int *omega, unsigned int log_n, FIELD field, const unsigned int * zeta, unsigned int ** dev_ptr, unsigned int start_n, void ** stream);