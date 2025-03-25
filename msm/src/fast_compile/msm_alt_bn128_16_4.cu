#include "../msm_impl.cuh"
#include "../alt_bn128.cuh"

namespace msm
{
    using ConfigG1 = MsmConfig<255, 16, 2, false>;
    template class MSM<ConfigG1, alt_bn128_g1::Number, alt_bn128_g1::Point, alt_bn128_g1::PointAffine>;
    template class MSMPrecompute<ConfigG1, alt_bn128_g1::Point, alt_bn128_g1::PointAffine>;
    template class MultiGPUMSM<ConfigG1, alt_bn128_g1::Number, alt_bn128_g1::Point, alt_bn128_g1::PointAffine>;

    using ConfigG2 = MsmConfig<255, 16, 2, false>;
    template class MSM<ConfigG2, alt_bn128_g2::Number, alt_bn128_g2::Point, alt_bn128_g2::PointAffine>;
    template class MSMPrecompute<ConfigG2, alt_bn128_g2::Point, alt_bn128_g2::PointAffine>;
    template class MultiGPUMSM<ConfigG2, alt_bn128_g2::Number, alt_bn128_g2::Point, alt_bn128_g2::PointAffine>;
}
