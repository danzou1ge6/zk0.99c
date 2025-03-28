#include "../msm_impl.cuh"
#include "../bn254.cuh"

namespace msm {
    using Config = MsmConfig<255, 16, 8, false>;
    using bn254_scalar::Params;
    using Number = mont::Number<8>;
    template class MSM<Config, Number, bn254::Point, bn254::PointAffine, bn254::PointAll>;
    template class MSMPrecompute<Config, bn254::Point, bn254::PointAffine>;
    template class MultiGPUMSM<Config, Number, bn254::Point, bn254::PointAffine, bn254::PointAll>;
}