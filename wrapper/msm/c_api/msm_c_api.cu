#include "./msm_c_api.h"
#include "../../../msm/src/msm_radix_sort.cuh"
#include "../../../msm/src/bn254.cuh"
#include "../../../mont/src/bn254_scalar.cuh"

#include <cuda_runtime.h>
#include <cub/cub.cuh>

using mont::u32;
using bn254::Point;
using bn254::PointAffine;
using bn254_scalar::Element;


bool cuda_msm(unsigned int len, const unsigned int* scalers, const unsigned int* points, unsigned int* res) {

    bool success = true;
    
    cudaHostRegister((void*)scalers, len * sizeof(Element), cudaHostRegisterDefault);
    cudaHostRegister((void*)points, len * sizeof(PointAffine), cudaHostRegisterDefault);

    u32 *d_points, *h_points_precompute, head;

    using Config = msm::MsmConfig<255, 22, 1, 1, 2, 2, true>;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    msm::precompute<Config>((u32*)points, len, h_points_precompute, head, stream);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    float elapsedTime = 0.0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    Point r;
    msm::run<Config>(len, (u32*)scalers, h_points_precompute, r, false, false, d_points, head, stream);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaStreamDestroy(stream);

    std::cout << "Total cost time:" << elapsedTime << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaHostUnregister((void*)scalers);
    cudaHostUnregister((void*)points);
    cudaFreeHost(h_points_precompute);

    auto r_affine = r.to_affine();

    for(int i=0;i<Element::LIMBS;++i) {
        res[i] = r_affine.x.n.limbs[i];
    }
    for(int i = 0; i < Element::LIMBS; ++i) {
        res[i+Element::LIMBS] = r_affine.y.n.limbs[i];
    }
    for(int i = 0; i < Element::LIMBS; ++i) {
        res[i + Element::LIMBS * 2] = Element::one().n.limbs[i];
    }

    return success;
}