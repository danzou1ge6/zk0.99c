#include "../../../msm/src/bn254.cuh"
#include "../../../msm/src/msm.cuh"
#include "../../../mont/src/bn254_scalar.cuh"

#include <iostream>
#include <fstream>

using bn254::Point;
using bn254::PointAffine;
using bn254_scalar::Element;
using bn254_scalar::Number;
using mont::u32;
using mont::u64;

struct MsmProblem
{
  u32 len;
  PointAffine *points;
  Element *scalers;
};

std::istream &
operator>>(std::istream &is, MsmProblem &msm)
{
  is >> msm.len;
  msm.scalers = new Element[msm.len];
  msm.points = new PointAffine[msm.len];
  for (u32 i = 0; i < msm.len; i++)
  {
    char _;
    is >> msm.scalers[i].n >> _ >> msm.points[i];
  }
  return is;
}

std::ostream &
operator<<(std::ostream &os, const MsmProblem &msm)
{

  for (u32 i = 0; i < msm.len; i++)
  {
    os << msm.scalers[i].n << '|' << msm.points[i] << std::endl;
  }
  return os;
}

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