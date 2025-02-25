#include "./msm_c_api.h"
#include "../../../msm/src/msm.cuh"
#include "../../../msm/src/bn254.cuh"
#include "../../../mont/src/bn254_scalar.cuh"

#include <cuda_runtime.h>
#include <cub/cub.cuh>

using mont::u32;
using bn254::Point;
using bn254::PointAffine;
using bn254_scalar::Number;
using bn254_scalar::Element;


bool cuda_msm(unsigned int len, const unsigned int* scalers, const unsigned int* points, unsigned int* res) {

    bool success = true;
    
    cudaHostRegister((void*)scalers, len * sizeof(Number), cudaHostRegisterDefault);
    cudaHostRegister((void*)points, len * sizeof(PointAffine), cudaHostRegisterDefault);

    using Config = msm::MsmConfig<255, 16, 16, false>;
    u32 batch_size = 1;
    u32 batch_per_run = 1;
    u32 parts = 2;
    u32 stage_scalers = 2;
    u32 stage_points = 2;

    std::array<u32*, Config::n_precompute> h_points;
    h_points[0] = (u32*)points;
    for (u32 i = 1; i < Config::n_precompute; i++) {
        cudaHostAlloc(&h_points[i], len * sizeof(PointAffine), cudaHostAllocDefault);
    }

    
    std::vector<u32*> scalers_batches;
    for (int i = 0; i < batch_size; i++) {
        scalers_batches.push_back((u32*)scalers);
    }

    std::vector<Point> r(batch_size);

    std::vector<u32> cards;
    int card_count;
    cudaGetDeviceCount(&card_count);
    for (int i = 0; i < card_count; i++) {
        cards.push_back(i);
    }

    msm::MultiGPUMSM<Config, Number, Point, PointAffine> msm_solver(len, batch_per_run, parts, stage_scalers, stage_points, cards);

    // std::cout << "start precompute" << std::endl;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    msm::MSMPrecompute<Config, Point, PointAffine>::precompute(len, h_points);
    msm_solver.set_points(h_points);

    // std::cout << "Precompute done" << std::endl;
    msm_solver.alloc_gpu();
    // std::cout << "Alloc GPU done" << std::endl;
    cudaEvent_t start, stop;
    float elapsedTime = 0.0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    msm_solver.msm(scalers_batches, r);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    // std::cout << "Run done" << std::endl;

    cudaStreamDestroy(stream);

    // for (int i = 0; i < batch_size; i++) {
    //     std::cout << r[i].to_affine() << std::endl;
    // }

    // std::cout << "Total cost time:" << elapsedTime << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaHostUnregister((void*)scalers);
    cudaHostUnregister((void*)points);
    for (u32 i = 1; i < Config::n_precompute; i++) {
        cudaFreeHost(h_points[i]);
    }

    auto r_affine = r[0].to_affine();

    auto x = r_affine.x;
    auto y = r_affine.y;
    auto z = Element::one();

    if (r_affine.is_identity()) { // identity
        x = Element::zero();
        y = Element::one();
        z = Element::zero();
    }

    for(int i=0;i<Element::LIMBS;++i) {
        res[i] = x.n.limbs[i];
    }
    for(int i = 0; i < Element::LIMBS; ++i) {
        res[i+Element::LIMBS] = y.n.limbs[i];
    }
    for(int i = 0; i < Element::LIMBS; ++i) {
        res[i + Element::LIMBS * 2] = z.n.limbs[i];
    }

    return success;
}