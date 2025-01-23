#pragma once
#include "../../mont/src/field.cuh"
#include "poly_eval.cuh"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#define PROPAGATE_CUDA_ERROR(x)                                                                                    \
  {                                                                                                                \
    auto err = x;                                                                                                       \
    if (err != cudaSuccess)                                                                                        \
    {                                                                                                              \
      std::cerr << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: " << cudaGetErrorString(err) << std::endl; \
      return err;                                                                                                  \
    }                                                                                                              \
  }

namespace poly{
using mont::u32;
using mont::u64;
using mont::usize;

// ground truth
// calculating q(x) = p(x) / (x - b)
// len(p) = len(res) + 1
template <typename Field>
void kate_divison(u32 len_p, Field *p, Field b, Field *q) {
    b = Field::zero()-b;
    Field tmp = Field::zero();
    q[len_p - 1] = Field::zero();
    for (long long i = len_p - 2; i >= 0; i--) {
        q[i] = p[i + 1] - tmp;
        tmp = q[i] * b;
    }
}

// recursive version for future paralleliza on GPU
// q[0]                           q[1]                     q[2]               q[3]
// p[1] + b * p[2] + b^2 * p[3]   p[2] + b * p[3]        | p[3]               0
// let sum = p[3] * b + p[2] = q[2] * b + p[2]
// p[1] + sum * b                 0 + sum                | p[3]               0
// neglect the sum, left and right are the answer of the subproblem
template <typename Field>
void recrusive_kate_divison(u32 len_p, Field *p, Field b, Field *q) {
    if (len_p == 1) {
        q[0] = Field::zero();
        return;
    }
    u32 mid = len_p / 2;
    recrusive_kate_divison(mid, p, b, q);
    recrusive_kate_divison(len_p - mid, p + mid, b, q + mid);
    Field sum = b * q[mid] + p[mid];
    for (int i = mid - 1; i >= 0; i--) {
        q[i] = q[i] + sum;
        sum = sum * b;
    }
}

template <typename Field>
__global__ void kate_kernel(Field *p, Field *q, Field *pow_b, Field b, u32 len_p, u32 deg) {
    u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len_p / 2) return;
    u32 seg_len = 1 << deg;
    u32 inter_seg_id = index / seg_len;
    u32 intra_seg_id = index % seg_len;
    u32 seg_start = inter_seg_id * seg_len * 2;
    u32 sum_id = seg_start + seg_len;
    q[seg_start + intra_seg_id] = q[seg_start + intra_seg_id] + (q[sum_id] * b + p[sum_id]) * pow_b[seg_len - 1 - intra_seg_id];
}

// assume len_p = 2^k
template <typename Field>
void gpu_kate_division(u32 log_p, Field *p, Field b, Field *q) {
    u32 len_p = 1 << log_p;
    u32 *pow_b;
    cudaMalloc(&pow_b, len_p / 2 * Field::LIMBS * sizeof(u32));
    get_pow_series<Field>(pow_b, b, len_p / 2, 0);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    cudaMemset(q, 0, len_p * Field::LIMBS * sizeof(u32));
    
    for (int deg = 0; deg < log_p; deg++) {
        u32 threads = 256;
        u32 blocks = (len_p / 2 + threads - 1) / threads;
        kate_kernel<Field> <<<blocks, threads>>> (p, q, reinterpret_cast<Field *>(pow_b), b, len_p, deg);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "gpu kate division: " << milliseconds << "ms" << std::endl;


    cudaFree(pow_b);
}

} // namespace poly