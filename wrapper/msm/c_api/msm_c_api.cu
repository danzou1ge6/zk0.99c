#include "./msm_c_api.h"

using mont::u32;

struct MsmProblem
{
  u32 len;
  PointAffine *points;
  Element *scalers;
};


bool cuda_msm(unsigned int len, Element* scalers, PointAffine* points, Point& res) {

    bool success = true;

    MsmProblem msm;

    msm.len = len;
    msm.scalers = scalers;
    msm.points = points;

    msm::run<msm::MsmConfig>((u32*)msm.scalers, (u32*)msm.points, msm.len, res);

    return success;
}

// bool cuda_coeff_to_extended(unsigned int *data, const unsigned int *omega, unsigned int log_n, FIELD field, const unsigned int * zeta, unsigned int **dev_ptr, unsigned int start_n, void ** stream) {
//     auto id = runtime::ntt_id{log_n, field, true, false};

//     bool success = true;
//     cudaError_t first_err = cudaSuccess;
//     cudaStream_t *stm;
//     stm = (cudaStream_t *)malloc(sizeof(cudaStream_t));
//     *stream = stm;
//     CUDA_CHECK(cudaStreamCreate((cudaStream_t *) stm));
//     if (field == FIELD::PASTA_CURVES_FIELDS_FP) {
//         CUDA_CHECK(cudaHostRegister((void *)data, 8ull * sizeof(uint) * (start_n), cudaHostRegisterDefault));
//     } else if (field == FIELD::HALO2CURVES_BN256_FR) {
//         CUDA_CHECK(cudaHostRegister((void *)data, 8ull * sizeof(uint) * (start_n), cudaHostRegisterDefault));
//     } else {
//         return false;
//     }
    
//     if (first_err == cudaSuccess) try {
//         auto ntt_kernel = temp_runtime.get_ntt_kernel(id, omega, nullptr, zeta);
//         CUDA_CHECK(ntt_kernel->ntt(data, *stm, start_n, dev_ptr));
//     } catch(const char *msg) {
//         std::cerr << msg << std::endl;
//         success = false;
//     }

//     return success;
// }