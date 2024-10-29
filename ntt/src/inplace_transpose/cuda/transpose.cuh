#pragma once
#include "../common/gcd.h"
#include "introspect.cuh"
#include "rotate.cuh"
#include "permute.cuh"
#include "equations.cuh"
#include "skinny.cuh"
#include "smem.cuh"
#include "memory_ops.cuh"
#include "util.cuh"
#include "register_ops.cuh"
#include <algorithm>

namespace inplace {

namespace c2r {
template <typename T>
void transpose(bool row_major, T* data, int m, int n, cudaStream_t stream = 0);
}
namespace r2c {
template <typename T>
void transpose(bool row_major, T* data, int m, int n, cudaStream_t stream = 0);
}

template <typename T>
void transpose(bool row_major, T* data, int m, int n, cudaStream_t stream = 0);

}



// implementation
namespace inplace {
namespace detail {

// these arch are too old, so we just disable them

// template<typename F>
// void sm_35_enact(double* data, int m, int n, F s) {
//     if (n < 3072) {
//         int smem_bytes = sizeof(double) * n;
//         smem_row_shuffle<<<m, 256, smem_bytes>>>(m, n, data, s);
//         check_error("smem shuffle");
//     } else if (n < 4100) {
//         register_row_shuffle<double, F, 16>
//             <<<m, 512>>>(m, n, data, s);
//         check_error("register 16 shuffle");
        
//     } else if (n < 6918) {
//         register_row_shuffle<double, F, 18>
//             <<<m, 512>>>(m, n, data, s);
//         check_error("register 18 shuffle");
        
//     } else if (n < 30208) {
//         register_row_shuffle<double, F, 59>
//             <<<m, 512>>>(m, n, data, s);
//         check_error("register 60 shuffle");
        
//     } else {
//         double* temp;
//         cudaMalloc(&temp, sizeof(double) * n * n_ctas());
//         memory_row_shuffle
//             <<<n_ctas(), n_threads()>>>(m, n, data, temp, s);
//         cudaFree(temp);
//         check_error("memory shuffle");
        
//     }
// }

// template<typename F>
// void sm_35_enact(float* data, int m, int n, F s) {
    
//     if (n < 6144) {
//         int smem_bytes = sizeof(float) * n;
//         smem_row_shuffle<<<m, 256, smem_bytes>>>(m, n, data, s);
//         check_error("smem shuffle");
//     } else if (n < 11326) {
//         register_row_shuffle<float, F, 31>
//             <<<m, 512>>>(m, n, data, s);
//         check_error("register 31 shuffle");
        
//     } else if (n < 30720) {
//         register_row_shuffle<float, F, 60>
//             <<<m, 512>>>(m, n, data, s);
//         check_error("register 60 shuffle");
        
//     } else {
//         float* temp;
//         cudaMalloc(&temp, sizeof(float) * n * n_ctas());
//         memory_row_shuffle
//             <<<n_ctas(), n_threads()>>>(m, n, data, temp, s);
//         cudaFree(temp);
//         check_error("memory shuffle");
        
//     }
// }

// template<typename F>
// void sm_52_enact(double* data, int m, int n, F s) {
//     if (n < 6144) {
//         int smem_bytes = sizeof(double) * n;
//         smem_row_shuffle<<<m, 256, smem_bytes>>>(m, n, data, s);
//         check_error("smem shuffle");
//     } else if (n < 6918) {
//         register_row_shuffle<double, F, 18>
//             <<<m, 512>>>(m, n, data, s);
//         check_error("register 18 shuffle");
        
//     } else if (n < 29696) {
//         register_row_shuffle<double, F, 57>
//             <<<m, 512>>>(m, n, data, s);
//         check_error("register 58 shuffle");
        
//     } else {
//         double* temp;
//         cudaMalloc(&temp, sizeof(double) * n * n_ctas());
//         memory_row_shuffle
//             <<<n_ctas(), n_threads()>>>(m, n, data, temp, s);
//         cudaFree(temp);
//         check_error("memory shuffle");
        
//     }
// }

// template<typename F>
// void sm_52_enact(float* data, int m, int n, F s) {
    
//     if (n < 12288) {
//         int smem_bytes = sizeof(float) * n;
//         smem_row_shuffle<<<m, 256, smem_bytes>>>(m, n, data, s);
//         check_error("smem shuffle");
//     } else if (n < 30720) {
//         register_row_shuffle<float, F, 60>
//             <<<m, 512>>>(m, n, data, s);
//         check_error("register 60 shuffle");
        
//     } else {
//         float* temp;
//         cudaMalloc(&temp, sizeof(float) * n * n_ctas());
//         memory_row_shuffle
//             <<<n_ctas(), n_threads()>>>(m, n, data, temp, s);
//         cudaFree(temp);
//         check_error("memory shuffle");
        
//     }
// }

template <typename T, typename F>
void sm_80_enact(T* data, int m, int n, F s, cudaStream_t stream) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    auto reserved_smem = prop.reservedSharedMemPerBlock;
    if (n * sizeof(T) < 167936 - reserved_smem) {
        int smem_bytes = sizeof(T) * n;
        cudaFuncSetAttribute(smem_row_shuffle<T, F>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        check_error("set attr");
        smem_row_shuffle<<<m, 256, smem_bytes, stream>>>(m, n, data, s);
        check_error("smem shuffle");
    } else if (n * sizeof(T) < 120 * 4 * 512) {
        register_row_shuffle<T, F, 120 * 4 / sizeof(T)>
            <<<m, 512, 0, stream>>>(m, n, data, s);
        check_error("register 120 shuffle");
        
    } else {
        T* temp;
        cudaMallocAsync(&temp, sizeof(T) * n * n_ctas(), stream);
        memory_row_shuffle
            <<<n_ctas(), n_threads(), 0, stream>>>(m, n, data, temp, s);
        cudaFreeAsync(temp, stream);
        check_error("memory shuffle");
    }
}

template <typename T, typename F>
void sm_86_enact(T* data, int m, int n, F s, cudaStream_t stream) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    auto reserved_smem = prop.reservedSharedMemPerBlock;
    if (n * sizeof(T) < 102400 - reserved_smem) {
        int smem_bytes = sizeof(T) * n;
        cudaFuncSetAttribute(smem_row_shuffle<T, F>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        check_error("set attr");
        smem_row_shuffle<<<m, 256, smem_bytes, stream>>>(m, n, data, s);
        check_error("smem shuffle");
    } else if (n * sizeof(T) < 60 * 4 * 512) {
        register_row_shuffle<T, F, 60 * 4 / sizeof(T)>
            <<<m, 512, 0, stream>>>(m, n, data, s);
        check_error("register 60 shuffle");

    } else if (n * sizeof(T) < 120 * 4 * 512) {
        register_row_shuffle<T, F, 120 * 4 / sizeof(T)>
            <<<m, 512, 0, stream>>>(m, n, data, s);
        check_error("register 120 shuffle");
    } else {
        T* temp;
        cudaMallocAsync(&temp, sizeof(T) * n * n_ctas(), stream);
        memory_row_shuffle
            <<<n_ctas(), n_threads(), 0, stream>>>(m, n, data, temp, s);
        cudaFreeAsync(temp, stream);
        check_error("memory shuffle");
    }
}

template<typename T, typename F>
void shuffle_fn(T* data, int m, int n, F s, cudaStream_t stream) {
    int arch = current_sm();
    if (arch == 800 || arch == 807) {
        sm_80_enact(data, m, n, s, stream);
    } else if (arch > 800) {
        sm_86_enact(data, m, n, s, stream);
    } else {
        T* temp;
        cudaMallocAsync(&temp, sizeof(T) * n * n_ctas(), stream);
        memory_row_shuffle
            <<<n_ctas(), n_threads(), 0, stream>>>(m, n, data, temp, s);
        cudaFreeAsync(temp, stream);
        check_error("memory shuffle");
    }
}

}

namespace c2r {

template<typename T>
void transpose(bool row_major, T* data, int m, int n, cudaStream_t stream) {
    if (!row_major) {
        std::swap(m, n);
    }
    //std::cout << "Doing C2R transpose of " << m << ", " << n << std::endl;

    int c, t, k;
    extended_gcd(m, n, c, t);
    if (c > 1) {
        extended_gcd(m/c, n/c, t, k);
    } else {
        k = t;
    }
    if (c > 1) {
        detail::rotate(detail::c2r::prerotator(n/c), m, n, data, stream);
    }
    detail::shuffle_fn(data, m, n, detail::c2r::shuffle(m, n, c, k), stream);
    detail::rotate(detail::c2r::postrotator(m), m, n, data, stream);
    int* temp_int;
    cudaMallocAsync(&temp_int, sizeof(int) * m, stream);
    detail::scatter_permute(detail::c2r::scatter_postpermuter(m, n, c), m, n, data, temp_int, stream);
    cudaFreeAsync(temp_int, stream);
}

}

namespace r2c {

template<typename T>
void transpose(bool row_major, T* data, int m, int n, cudaStream_t stream) {
    if (row_major) {
        std::swap(m, n);
    }
    //std::cout << "Doing R2C transpose of " << m << ", " << n << std::endl;

    int c, t, k;
    extended_gcd(m, n, c, t);
    if (c > 1) {
        extended_gcd(m/c, n/c, t, k);
    } else {
        k = t;
    }
    int* temp_int;
    cudaMallocAsync(&temp_int, sizeof(int) * m, stream);
    detail::scatter_permute(detail::r2c::scatter_prepermuter(m, n, c), m, n, data, temp_int, stream);
    cudaFreeAsync(temp_int, stream);
    detail::rotate(detail::r2c::prerotator(m), m, n, data, stream);
    detail::shuffle_fn(data, m, n, detail::r2c::shuffle(m, n, c, k), stream);
    if (c > 1) {
        detail::rotate(detail::r2c::postrotator(n/c, m), m, n, data, stream);
    }
}

}


template<typename T>
void transpose(bool row_major, T* data, int m, int n, cudaStream_t stream) {
    bool small_m = m < 32;
    bool small_n = n < 32;
    //Heuristic to choose the fastest implementation
    //based on size of matrix and data layout
    if (!small_m && small_n) {
        std::swap(m, n);
        if (!row_major) {
            inplace::detail::c2r::skinny_transpose(
                data, m, n, stream);
        } else {
            inplace::detail::r2c::skinny_transpose(
                data, m, n, stream);
        }
    } else if (small_m) {
        if (!row_major) {
            inplace::detail::r2c::skinny_transpose(
                data, m, n, stream);
        } else {
            inplace::detail::c2r::skinny_transpose(
                data, m, n, stream);
        }
    } else {
        bool m_greater = m > n;
        if (m_greater ^ row_major) {
            inplace::r2c::transpose(row_major, data, m, n, stream);
        } else {
            inplace::c2r::transpose(row_major, data, m, n, stream);
        }
    }
}

}
