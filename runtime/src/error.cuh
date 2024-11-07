#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

#define CUDA_CHECK_EXIT(call)                                                                                        \
{                                                                                                                    \
    cudaError_t err = call;                                                                                          \
    if (err != cudaSuccess) {                                                                                        \
        std::cerr << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: " << cudaGetErrorString(err) << std::endl; \
        exit(1);                                                                                                     \
    }                                                                                                                \
}

#define CUDA_CHECK_THROW(call)                                                                                        \
{                                                                                                                    \
    cudaError_t err = call;                                                                                          \
    if (err != cudaSuccess) {                                                                                        \
        throw std::runtime_error("CUDA Error [" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]: " + cudaGetErrorString(err)); \
    }                                                                                                                \
}
