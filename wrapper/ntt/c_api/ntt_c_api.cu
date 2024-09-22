#include "../../../ntt/src/self_sort_in_place_ntt.cuh"
#include "./ntt_c_api.h"
#include "../../../ntt/src/scheduler.cuh"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

bool cuda_ntt(unsigned int *data, const unsigned int *omega, unsigned int log_n, FIELD field) {
    scheduler::ntt_id id;
    id.field = field;
    id.log_len = log_n;
    if (field == FIELD::PASTA_CURVES_FIELDS_FP) {
        for (int i = 0; i < 8; i++) id.omega.push_back(omega[i]);
    } else if (field == FIELD::HALO2CURVES_BN256_FR) {
        for (int i = 0; i < 8; i++) id.omega.push_back(omega[i]);
    } else {
        return false;
    }
    
    auto ntt_kernel = scheduler::get_ntt_kernel(id);
    auto err = ntt_kernel->ntt(data);

    return err == cudaSuccess;
}  

#ifdef __cplusplus
}
#endif