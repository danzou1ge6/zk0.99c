#include "./msm_c_api.h"
#include "../../../msm/src/msm.cuh"
#include "../../../msm/src/bn254.cuh"
#include "../../../mont/src/bn254_scalar.cuh"

#include <cuda_runtime.h>
#include <cub/cub.cuh>

using mont::u32;
using bn254::Point;
using bn254::PointAffine;
using bn254::PointAll;
using bn254::PointAffineAll;
using bn254_scalar::Params;
using Element = mont::Element<Params, Params::LIMBS>;

// void rearrange(u32* arr, u32 len) {
//     for(u32 j=0; j<len; ++j) {
//         u32 block_size = Params::LIMBS / TPI;
//         u32 total_elements = 2 * Params::LIMBS;
//         u32* temp = (u32*)malloc(total_elements * sizeof(u32));
        
//         for (u32 i = 0; i < TPI; ++i) {
//             u32 x_start = i * block_size;
//             u32 y_start = Params::LIMBS + i * block_size;
//             u32 temp_start = i * 2 * block_size;
            
//             // 复制x的当前块
//             memcpy(temp + temp_start, arr + x_start, block_size * sizeof(u32));
//             // 复制y的当前块
//             memcpy(temp + temp_start + block_size, arr + y_start, block_size * sizeof(u32));
//         }
        
//         // 将结果复制回原数组
//         memcpy(arr, temp, total_elements * sizeof(u32));
//     }
//     free(temp);
// }

bool cuda_msm(unsigned int len, const unsigned int* scalers, const unsigned int* points, unsigned int* res) {

    bool success = true;
    
    cudaHostRegister((void*)scalers, len * sizeof(Element), cudaHostRegisterDefault);
    cudaHostRegister((void*)points, len * sizeof(PointAffine) * TPI, cudaHostRegisterDefault);

    using Config = msm::MsmConfig<255, 16, 16, false>;
    u32 batch_size = 1;
    u32 batch_per_run = 1;
    u32 parts = 2;
    u32 stage_scalers = 2;
    u32 stage_points = 2;

    std::array<u32*, Config::n_precompute> h_points;
    h_points[0] = (u32*)points;
    for (u32 i = 1; i < Config::n_precompute; i++) {
        cudaHostAlloc(&h_points[i], len * sizeof(PointAffine) * TPI, cudaHostAllocDefault);
    }

    
    std::vector<u32*> scalers_batches;
    for (int i = 0; i < batch_size; i++) {
        scalers_batches.push_back((u32*)scalers);
    }

    std::vector<PointAll> r(batch_size);

    std::vector<u32> cards;
    int card_count;
    cudaGetDeviceCount(&card_count);
    for (int i = 0; i < card_count; i++) {
        cards.push_back(i);
    }

    msm::MultiGPUMSM<Config, Element, Point, PointAffine, PointAll> msm_solver(len, batch_per_run, parts, stage_scalers, stage_points, cards);

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
        res[i] = x.n._limbs[i];
    }
    for(int i = 0; i < Element::LIMBS; ++i) {
        res[i+Element::LIMBS] = y.n._limbs[i];
    }
    for(int i = 0; i < Element::LIMBS; ++i) {
        res[i + Element::LIMBS * 2] = z.n._limbs[i];
    }

    return success;
}