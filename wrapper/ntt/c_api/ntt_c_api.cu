#include "./ntt_c_api.h"
#include "../../../ntt/src/runtime.cuh"
#include <cuda_runtime.h>

runtime::ntt_runtime<runtime::fifo> temp_runtime(1);

bool cuda_ntt(unsigned int *data, const unsigned int *omega, unsigned int log_n, FIELD field, bool inverse, bool process, const unsigned int * inv_n, const unsigned int * zeta) {
    runtime::ntt_id id;
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
    
    if (first_err == cudaSuccess) try {
        auto ntt_kernel = temp_runtime.get_ntt_kernel(id);
        CUDA_CHECK(ntt_kernel->ntt(data, inverse, process, inv_n, zeta));
    } catch(const char *msg) {
        std::cerr << msg << std::endl;
        success = false;
    }
    CUDA_CHECK(cudaHostUnregister((void *)data));

    return success;
}