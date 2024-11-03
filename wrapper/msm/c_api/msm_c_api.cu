#include "./msm_c_api.h"
#include "../../../msm/src/msm.cuh"

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