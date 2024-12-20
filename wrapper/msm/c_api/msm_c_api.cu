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

    printf("Host register done\n");

    using Config = msm::MsmConfig<255, 22, 1, false>;
    u32 batch_size = 4;
    u32 batch_per_run = 2;
    u32 parts = 8;
    u32 stage_scalers = 2;
    u32 stage_points = 2;

    std::array<u32*, Config::n_precompute> h_points;
    for (u32 i = 0; i < Config::n_precompute; i++) {
        cudaHostAlloc(&h_points[i], len * sizeof(PointAffine), cudaHostAllocDefault);
    }
    memcpy((void*)h_points[0], (void*)points, len * sizeof(PointAffine));

    printf("host alloc points done\n");

    const u32 **scaler_batches = new const u32*[batch_size];
    scaler_batches[0] = (u32*)scalers;

    for (int i = 1; i < batch_size; i++) {
        cudaHostAlloc(&scaler_batches[i], len * sizeof(Element), cudaHostAllocDefault);
        memcpy((void*)scaler_batches[i], (void*)scalers, len * sizeof(Element));
    }

    printf("host alloc scalers done\n");
    Point *r = new Point[batch_size];

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    msm::MSM<Config> msm_solver(len, batch_per_run, parts, stage_scalers, stage_points);

    printf("start precompute\n");

    msm_solver.precompute(std::move(h_points), stream);

    printf("Precompute done\n");

    msm_solver.alloc_gpu();

    cudaEvent_t start, stop;
    float elapsedTime = 0.0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    msm_solver.msm(batch_size, scaler_batches, r, stream);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Run done\n");

    for (int i = 1; i < batch_size; i++) {
        cudaFreeHost((void*)scaler_batches[i]);
    }

    delete [] scaler_batches;

    cudaStreamDestroy(stream);

    std::cout << "Total cost time:" << elapsedTime << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaHostUnregister((void*)scalers);

    auto r_affine = r[0].to_affine();

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