#pragma once
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK_EXIT(call)                                                                                        \
{                                                                                                                    \
    cudaError_t err = call;                                                                                          \
    if (err != cudaSuccess) {                                                                                        \
        std::cerr << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: " << cudaGetErrorString(err) << std::endl; \
        exit(1);                                                                                                     \
    }                                                                                                                \
}
