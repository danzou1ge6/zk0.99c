#pragma once

#include "../common/index.cuh"
#include <set>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include "util.cuh"

namespace inplace {
namespace detail {

template<typename T, typename F>
void scatter_permute(F f, int m, int n, T* data, int* tmp, cudaStream_t stream = 0);

}
}

namespace inplace {
namespace detail {
    
template<typename Fn>
void scatter_cycles(Fn f, int *heads, int &heads_sz, int *lens, int &lens_sz) {
    int len = f.len();
    thrust::counting_iterator<int> i(0);
    std::set<int> unvisited(i, i+len);
    while(!unvisited.empty()) {
        int idx = *unvisited.begin();
        unvisited.erase(unvisited.begin());
        int dest = f(idx);
        if (idx != dest) {
            heads[heads_sz++] = idx;
            int start = idx;
            int len = 1;
            //std::cout << "Cycle: " << start << " " << dest << " ";
            while(dest != start) {
                idx = dest;
                unvisited.erase(idx);
                dest = f(idx);
                len++;
                //std::cout << dest << " ";
            }
            //std::cout << std::endl;
            lens[lens_sz++] = len;
        }
    }
}


template<typename T, typename F, int U>
__device__ __forceinline__ void unroll_cycle_row_permute(
    F f, row_major_index rm, T* data, int i, int j, int l) {
    
    T src = data[rm(i, j)];
    T loaded[U+1];
    loaded[0] = src;
    for(int k = 0; k < l / U; k++) {
        int rows[U];
#pragma unroll
        for(int x = 0; x < U; x++) {
            i = f(i);
            rows[x] = i;
        }
#pragma unroll
        for(int x = 0; x < U; x++) {
            loaded[x+1] = data[rm(rows[x], j)];
        }
#pragma unroll
        for(int x = 0; x < U; x++) {
            data[rm(rows[x], j)] = loaded[x];
        }
        loaded[0] = loaded[U];
    }
    T tmp = loaded[0];
    for(int k = 0; k < l % U; k++) {
        i = f(i);
        T new_tmp = data[rm(i, j)];
        data[rm(i, j)] = tmp;
        tmp = new_tmp;
    }
    
}

template<typename T, typename F, int U>
__global__ void cycle_row_permute(F f, T* data, int* heads,
                                  int* lens, int n_heads) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int n = f.n;
    row_major_index rm(f.m, f.n);


    if ((j < n) && (h < n_heads)) {
        int i = heads[h];
        int l = lens[h];
        unroll_cycle_row_permute<T, F, U>(f, rm, data, i, j, l);
    }
}

template<typename T, typename F>
void scatter_permute(F f, int m, int n, T* data, int* tmp, cudaStream_t stream) {
    int *heads, *lens;
    cudaHostAlloc(&heads, sizeof(int) * m / 2, cudaHostAllocDefault);
    cudaHostAlloc(&lens, sizeof(int) * m / 2, cudaHostAllocDefault);
    int heads_sz = 0, lens_sz = 0;

    scatter_cycles(f, heads, heads_sz, lens, lens_sz);
    int* d_heads = tmp;
    int* d_lens = tmp + m / 2;
    cudaMemcpyAsync(d_heads, heads, sizeof(int)*heads_sz,
               cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_lens, lens, sizeof(int)*lens_sz,
               cudaMemcpyHostToDevice, stream);
    int n_threads_x = 256;
    int n_threads_y = 1024/n_threads_x;
    
    int n_blocks_x = div_up(n, n_threads_x);
    int n_blocks_y = div_up(heads_sz, n_threads_y);
    cycle_row_permute<T, F, 4>
        <<<dim3(n_blocks_x, n_blocks_y),
        dim3(n_threads_x, n_threads_y), 0, stream>>>
        (f, data, d_heads, d_lens, heads_sz);

    cudaFreeHost(heads);
    cudaFreeHost(lens);
}

}
}

