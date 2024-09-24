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

    bool success = true;
    cudaError_t first_err = cudaSuccess;
    if (field == FIELD::PASTA_CURVES_FIELDS_FP) {
        CUDA_CHECK(cudaHostRegister((void *)data, 8ull * sizeof(uint) * (1 << log_n), cudaHostRegisterDefault));
        for (int i = 0; i < 8; i++) id.omega.push_back(omega[i]);
    } else if (field == FIELD::HALO2CURVES_BN256_FR) {
        CUDA_CHECK(cudaHostRegister((void *)data, 8ull * sizeof(uint) * (1 << log_n), cudaHostRegisterDefault));
        for (int i = 0; i < 8; i++) id.omega.push_back(omega[i]);
    } else {
        return false;
    }
    
    auto ntt_kernel = scheduler::get_ntt_kernel(id);
    CUDA_CHECK(ntt_kernel->ntt(data));
    CUDA_CHECK(cudaHostUnregister((void *)data));
    CUDA_CHECK(ntt_kernel->clean_gpu());

    return success;
}  

#ifdef __cplusplus
}
#endif