#include "../msm_impl.cuh"
#include "../bn254.cuh"

namespace msm {
    using Config = MsmConfig<255, 11, 32, false>;
    template class MSM<Config, bn254::Number, bn254::Point, bn254::PointAffine>;
    template class MSMPrecompute<Config, bn254::Point, bn254::PointAffine>;
    template class MultiGPUMSM<Config, bn254::Number, bn254::Point, bn254::PointAffine>;
}
