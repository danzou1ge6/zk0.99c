#include "./msm_c_api.h"
#include "../../../msm/src/msm.cuh"
#include "../../../msm/src/bn254.cuh"
#include "../../../mont/src/bn254_scalar.cuh"

#include <cuda_runtime.h>
#include <cub/cub.cuh>

using mont::u32;
using bn254::Point;
using bn254::PointAffine;
using bn254_scalar::Element;

struct MsmProblem
{
  u32 len;
  PointAffine *points;
  Element *scalers;
};


bool cuda_msm(unsigned int len, const unsigned int* scalers, const unsigned int* points, unsigned int* res) {

    bool success = true;

    MsmProblem msm;

    msm.len = len;

    Point result;

    msm::run<msm::MsmConfig>(scalers, points, msm.len, result);

    for(int i=0;i<8;++i)
    {
      res[i] = result.x.n.limbs[i];
    }
    for(int i=0;i<8;++i)
    {
      res[i+8] = result.y.n.limbs[i];
    }
    for(int i=0;i<8;++i)
    {
      res[i+16] = result.z.n.limbs[i];
    }

    return success;
}