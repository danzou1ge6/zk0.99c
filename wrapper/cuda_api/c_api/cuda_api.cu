#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include "cuda_api.h"

#define CUDA_CHECK(call)                                                                                             \
{                                                                                                                    \
    cudaError_t err = call;                                                                                          \
    if (err != cudaSuccess) {                                                                                        \
        std::cerr << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: " << cudaGetErrorString(err) << std::endl; \
        success = false;                                                                                             \
    }                                                                                                                \
}

bool cuda_device_to_host_sync(void *dst, const void *src, unsigned long size, const void* stream_ptr) {
    cudaStream_t stream =  stream_ptr == nullptr ? 0 : *(cudaStream_t*)stream_ptr;
    bool success = true;
    
    CUDA_CHECK(cudaHostRegister(dst, size, cudaHostRegisterDefault));
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaHostUnregister(dst));

    return success;
}

bool cuda_unregister(void *src) {
    bool success = true;
    CUDA_CHECK(cudaHostUnregister(src));
    return success;
}

void cpp_free(void * src) {
    free(src);
}

bool cuda_free(void * dev_ptr, const void *stream_ptr) {
    cudaStream_t stream =  stream_ptr == nullptr ? 0 : *(cudaStream_t*)stream_ptr;
    bool success = true;
    CUDA_CHECK(cudaFreeAsync(dev_ptr, stream));
    return success;
}