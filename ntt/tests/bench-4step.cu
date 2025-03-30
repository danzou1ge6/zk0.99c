#include <random>
#include <ctime>
#include <cuda_runtime.h>
#include "../src/4step_ntt.cuh"
#include "../../mont/src/bn254_fr.cuh"

typedef bn254_fr::Element Field;

int main() {
    for (int k = 20; k <= 30; k += 2) {
        auto omega = Field::host_random();

        long long length = 1ll << k;

        printf("k = %d, length = %lld\n", k, length);

        Field *src_gpu, *dst_gpu;
        cudaHostAlloc(&src_gpu, length * sizeof(Field), cudaHostAllocDefault);
        cudaHostAlloc(&dst_gpu, length * sizeof(Field), cudaHostAllocDefault);
        
        for (int i = 0; i < (1ll << k); i++) {
            src_gpu[i] = Field::host_random();
        }

        // warm up, because the jit compilation is slow
        ntt::offchip_ntt<Field>((uint*)src_gpu, (uint*)dst_gpu, k, (uint*)&omega);
        cudaFreeHost(dst_gpu);
        cudaFreeHost(src_gpu);
    }
    return 0;
}

