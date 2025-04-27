#include "../msm_impl.cuh"
#include "../bn254.cuh"

namespace msm {
    using Config = MsmConfig<255, 16, 16, false, TPI>;
    using Number = mont::Number<8>;
    template class MSM<Config, Number, bn254::Point, bn254::PointAffine, bn254::PointAll, bn254::PointAffineAll>;
    template class MSMPrecompute<Config, bn254::Point, bn254::PointAffine, bn254::PointAffineAll>;
    template class MultiGPUMSM<Config, Number, bn254::Point, bn254::PointAffine, bn254::PointAll, bn254::PointAffineAll>;
}