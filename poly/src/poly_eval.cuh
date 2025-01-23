#pragma once
#include "../../mont/src/field.cuh"
#include "poly.cuh"
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

template <typename Field>
__global__ void naive_kernel(const u32 *poly, u32 *temp_buf, Field x, u64 len) {
    u64 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    auto val = Field::load(poly + index * Field::LIMBS);
    val = val * x.pow(index);
    val.store(temp_buf + index * Field::LIMBS);
}

template <typename Field>
cudaError_t NaiveEval(const u32 *poly, u32* temp_buf, u32* res, Field x, u64 len, cudaStream_t stream) {
    auto threads = 256;
    auto blocks = (len + threads - 1) / threads;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    naive_kernel<Field><<<blocks, threads>>>(poly, temp_buf, x, len);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "naive_kernel: " << milliseconds << "ms" << std::endl;

    // ruduce all
    void *d_temp_storage_reduce = nullptr;
    usize temp_storage_bytes_reduce = 0;

    auto add_op = [] __device__ __host__(const Field &a, const Field &b) { return a + b; };

    PROPAGATE_CUDA_ERROR(
        cub::DeviceReduce::Reduce(
            d_temp_storage_reduce, 
            temp_storage_bytes_reduce, 
            reinterpret_cast<const Field*>(temp_buf), 
            reinterpret_cast<Field*>(res),
            len,
            add_op,
            Field::zero(),
            stream
        )
    );

    PROPAGATE_CUDA_ERROR(cudaMallocAsync(&d_temp_storage_reduce, temp_storage_bytes_reduce, stream));

    cudaEventRecord(start);
    PROPAGATE_CUDA_ERROR(
        cub::DeviceReduce::Reduce(
            d_temp_storage_reduce, 
            temp_storage_bytes_reduce, 
            reinterpret_cast<const Field*>(temp_buf), 
            reinterpret_cast<Field*>(res),
            len, 
            add_op,
            Field::zero(),
            stream
        )
    );
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "reduce: " << milliseconds << "ms" << std::endl;
    cudaFreeAsync(d_temp_storage_reduce, stream);
    return cudaSuccess;
}

template <typename Field>
__global__ void init_pow_series(u32 *temp_buf, Field x, u64 len) {
    u64 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    if (index == 0) {
        Field::one().store(temp_buf);
    } else {
        x.store(temp_buf + index * Field::LIMBS);
    }
}

template<typename Field>
cudaError_t get_pow_series(u32 *pow_series, Field x, u64 len, cudaStream_t stream) {
    u32 threads = 256;
    u32 blocks = (len + threads - 1) / threads;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    init_pow_series<Field><<<blocks, threads, 0, stream>>>(pow_series, x, len);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "init_pow_series: " << milliseconds << "ms" << std::endl;

    void *d_temp_scan = nullptr;
    usize temp_scan_size = 0;

    auto mul_op = [] __device__ __host__(const Field &a, const Field &b) { return a * b; };

    cub::DeviceScan::InclusiveScan(d_temp_scan, temp_scan_size, reinterpret_cast<Field*>(pow_series), mul_op, len, stream);
    cudaMallocAsync(&d_temp_scan, temp_scan_size, stream);

    cudaEventRecord(start);
    cub::DeviceScan::InclusiveScan(d_temp_scan, temp_scan_size, reinterpret_cast<Field*>(pow_series), mul_op, len, stream);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "scan: " << milliseconds << "ms" << std::endl;

    cudaFreeAsync(d_temp_scan, stream);
    return cudaSuccess;
}

template <typename Field>
cudaError_t Eval(const u32 *poly, u32* temp_buf, u32* res, Field x, u64 len, cudaStream_t stream) {
    get_pow_series<Field>(temp_buf, x, len, stream);
    u32 threads = 256;
    u32 blocks = (len + threads - 1) / threads;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    NaiveMul<Field><<<blocks, threads, 0, stream>>>(poly, temp_buf, temp_buf, len);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "mul: " << milliseconds << "ms" << std::endl;

    // ruduce all
    void *d_temp_storage_reduce = nullptr;
    usize temp_storage_bytes_reduce = 0;

    auto add_op = [] __device__ __host__(const Field &a, const Field &b) { return a + b; };

    PROPAGATE_CUDA_ERROR(
        cub::DeviceReduce::Reduce(
            d_temp_storage_reduce, 
            temp_storage_bytes_reduce, 
            reinterpret_cast<const Field*>(temp_buf), 
            reinterpret_cast<Field*>(res),
            len,
            add_op,
            Field::zero(),
            stream
        )
    );

    PROPAGATE_CUDA_ERROR(cudaMallocAsync(&d_temp_storage_reduce, temp_storage_bytes_reduce, stream));

    cudaEventRecord(start);
    PROPAGATE_CUDA_ERROR(
        cub::DeviceReduce::Reduce(
            d_temp_storage_reduce, 
            temp_storage_bytes_reduce, 
            reinterpret_cast<const Field*>(temp_buf), 
            reinterpret_cast<Field*>(res),
            len, 
            add_op,
            Field::zero(),
            stream
        )
    );
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "reduce: " << milliseconds << "ms" << std::endl;

    cudaFreeAsync(d_temp_storage_reduce, stream);
    return cudaSuccess;

}

} // namespace poly