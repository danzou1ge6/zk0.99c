#include "../msm_impl.cuh"
#include "../bls12381.cuh"

namespace msm {
    using Config = MsmConfig<255, 14, 19, false, TPI>;
    using Number = mont::Number<8>;
    template class MSM<Config, Number, bls12381::Point, bls12381::PointAffine, bls12381::PointAll, bls12381::PointAffineAll>;
    template class MSMPrecompute<Config, bls12381::Point, bls12381::PointAffine, bls12381::PointAffineAll>;
    template class MultiGPUMSM<Config, Number, bls12381::Point, bls12381::PointAffine, bls12381::PointAll, bls12381::PointAffineAll>;
}