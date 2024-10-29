#include "../src/inplace_transpose/cuda/transpose.cuh"
#include <cstring>
#include <cuda_runtime.h>
#define IDX(i, j, n) ((i) * (n) + (j))
struct chunk {
    int data[8];
    __device__ __host__ __forceinline__ chunk& operator=(const chunk& other) {
        #ifdef __CUDA_ARCH__
            reinterpret_cast<uint4*>(data)[0] = reinterpret_cast<const uint4*>(other.data)[0];
            reinterpret_cast<uint4*>(data)[1] = reinterpret_cast<const uint4*>(other.data)[1];
        #else
        for (int i = 0; i < 8; i++) {
            data[i] = other.data[i];
        }
        #endif
        return *this;
    }
};

int main() {
    chunk *data, *data_d, *data_h;
    int m = 16, n = (1 << 26) / m;
    cudaHostAlloc(&data, m * n * sizeof(chunk), cudaHostAllocDefault);
    cudaHostAlloc(&data_h, m * n * sizeof(chunk), cudaHostAllocDefault);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < 8; k++) {
                data[IDX(i, j, n)].data[k] = i * n + j + k;
            }
        }
    }
    memcpy(data_h, data, sizeof(chunk) * m * n);

    cudaMalloc(&data_d, sizeof(chunk) * m * n);
    cudaMemcpy(data_d, data, sizeof(chunk) * m * n, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    inplace::transpose(true, data_d, m, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %f ms\n", milliseconds);

    cudaMemcpy(data, data_d, sizeof(chunk) * m * n, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            for (int k = 0; k < 8; k++) {
                assert(data[IDX(i, j, m)].data[k] == data_h[IDX(j, i, n)].data[k]);
            }
        }
    }

    cudaFree(data_d);
    cudaFreeHost(data);
    return 0;
}