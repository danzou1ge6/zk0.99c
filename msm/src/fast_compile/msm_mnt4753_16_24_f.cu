#include "../msm_impl.cuh"
#include "../mnt4753.cuh"

namespace msm {
    using Config = MsmConfig<753, 16, 24, false, TPI>;
    using Number = mont::Number<24>;
    template class MSM<Config, Number, mnt4753::Point, mnt4753::PointAffine, mnt4753::PointAll, mnt4753::PointAffineAll>;
    template class MSMPrecompute<Config, mnt4753::Point, mnt4753::PointAffine, mnt4753::PointAffineAll>;
    template class MultiGPUMSM<Config, Number, mnt4753::Point, mnt4753::PointAffine, mnt4753::PointAll, mnt4753::PointAffineAll>;
}