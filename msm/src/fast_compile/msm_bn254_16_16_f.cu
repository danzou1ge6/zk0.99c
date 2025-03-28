#include "../msm_impl.cuh"
#include "../bn254.cuh"

namespace msm {
    using Config = MsmConfig<255, 16, 16, false>;
    using bn254_scalar::Params;
    using Element = mont::Element<Params, Params::LIMBS>;
    template class MSM<Config, Element, bn254::Point, bn254::PointAffine, bn254::PointAll>;
    template class MSMPrecompute<Config, bn254::Point, bn254::PointAffine>;
    template class MultiGPUMSM<Config, Element, bn254::Point, bn254::PointAffine, bn254::PointAll>;
}