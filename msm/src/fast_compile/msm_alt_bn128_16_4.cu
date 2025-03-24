#include "../msm_impl.cuh"
#include "../alt_bn128.cuh"

namespace alt_bn128_g1 {
    using namespace msm;

    using Config = MsmConfig<255, 16, 2, false>;
    template class MSM<Config, Number, Point, PointAffine>;
    template class MSMPrecompute<Config, Point, PointAffine>;
    template class MultiGPUMSM<Config, Number, Point, PointAffine>;
}


namespace alt_bn128_g2 {
    using namespace msm;

    using Config = MsmConfig<255, 16, 2, false>;
    template class MSM<Config, Number, Point, PointAffine>;
    template class MSMPrecompute<Config, Point, PointAffine>;
    template class MultiGPUMSM<Config, Number, Point, PointAffine>;
}
