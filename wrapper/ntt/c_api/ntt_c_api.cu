#include "./ntt_c_api.h"
#include "../../../ntt/src/runtime.cuh"
#include <cuda_runtime.h>

runtime::ntt_runtime<runtime::fifo> temp_runtime(1);

bool cuda_ntt(unsigned int *data, const unsigned int *omega, unsigned int log_n, FIELD field, bool inverse, bool process, const unsigned int * inv_n, const unsigned int * zeta) {
    auto id = runtime::ntt_id{log_n, field, process, inverse};

    bool success = true;
    cudaError_t first_err = cudaSuccess;
    if (field == FIELD::PASTA_CURVES_FIELDS_FP) {
        CUDA_CHECK(cudaHostRegister((void *)data, 8ull * sizeof(uint) * (1 << log_n), cudaHostRegisterDefault));
    } else if (field == FIELD::HALO2CURVES_BN256_FR) {
        CUDA_CHECK(cudaHostRegister((void *)data, 8ull * sizeof(uint) * (1 << log_n), cudaHostRegisterDefault));
    } else {
        return false;
    }
    
    if (first_err == cudaSuccess) try {
        auto ntt_kernel = temp_runtime.get_ntt_kernel(id, omega, inv_n, zeta);
        CUDA_CHECK(ntt_kernel->ntt(data));
    } catch(const char *msg) {
        std::cerr << msg << std::endl;
        success = false;
    }
    CUDA_CHECK(cudaHostUnregister((void *)data));

    return success;
}