#include "../src/recompute_ntt.cuh"
#include "../../mont/src/bn254_fr.cuh"
#include "../src/cooley_turkey_ntt.cuh"

using namespace ntt;
typedef bn254_fr::Element Field;

int main () {
    for (int k = 20; k <= 28; k += 2) {
        auto omega = Field::host_random();

        cooley_turkey_ntt<Field> ntt(reinterpret_cast<u32*>(&omega), k, false);
        ntt.to_gpu();
        
        Field *data, *data_d;
        cudaMalloc(&data_d, (1ll << k) * sizeof(Field));
        data = (Field*)malloc((1ll << k) * sizeof(Field));
        for (int i = 0; i < (1ll << k); i++) {
            data[i] = Field::host_random();
        }
        cudaMemcpy(data_d, data, (1ll << k) * sizeof(Field), cudaMemcpyHostToDevice);

        // warm up, because the jit compilation is slow
        // for (int i = 0; i < 10; i++) ntt.ntt(reinterpret_cast<u32*>(data_d), 0, 0, true);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (int i = 0; i < 20; i++) {
            cudaEventRecord(start);
            ntt.ntt(reinterpret_cast<u32*>(data_d), 0, 0, true);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("k = %d, time = %f ms\n", k, milliseconds);
        }
        
        free(data);
        cudaFree(data_d);
    }
    return 0;
}
