#pragma once

#include "../../mont/src/mont.cuh"
#include <cuda_runtime.h>

namespace NTT {
    typedef u_int32_t u32;
    
    template<u32 words>
    class naive_ntt {
        __global__ void rearrange(u32 * data, uint2 * reverse, u32 len) {
            u32 index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= len) return;
            uint2 r = reverse[index];
            
            #pragma unroll
            for (u32 i = 0; i < words; i++) {
                u32 tmp = data[r.x * words + i];
                data[r.x * words + i] = data[r.y * words + i];
                data[r.y * words + i] = tmp;
            }
        }
        
        u32 gen_reverse(u32 log_len, uint2* reverse_pair) {
            u32 len = 1 << log_len;
            u32* reverse = new u32[len];
            for (u32 i = 0; i < len; i++) {
                reverse[i] = (reverse[i >> 1] >> 1) | ((i & 1) << (log_len - 1) ); //reverse the bits
            }
            int r_len = 0;
            for (u32 i = 0; i < len; i++) {
                if (reverse[i] < i) {
                    reverse_pair[r_len].x = i;
                    reverse_pair[r_len].y = reverse[i];
                    r_len++;
                }
            }
            delete[] reverse;
            return r_len;
        }

        void gen_roots(u32 * roots, u32 len, u32 root) {
            
            roots[0] = 1;
            for (u32 i = 0; i < len * words; i+= words) {

            }
        }

        public:
        void ntt(u32 * data, u32 log_len) {
            u32 len = 1 << log_len;
            uint2 * reverse, * reverse_d;
            reverse = (uint2 *) malloc(len * sizeof(uint2));
            u32 r_len = gen_reverse(log_len, reverse);
            cudaMalloc(&reverse_d, r_len * sizeof(uint2));
            cudaMemcpy(reverse_d, reverse, r_len * sizeof(uint2), cudaMemcpyHostToDevice);
            
            dim3 rearrange_block(768);
            dim3 rearrange_grid((r_len + rearrange_block.x - 1) / rearrange_block.x);

            u32 * data_d;
            cudaMalloc(&data_d, len * words * sizeof(u32));
            cudaMemcpy(data_d, data, len * words * sizeof(u32), cudaMemcpyHostToDevice);

            rearrange<<<rearrange_grid, rearrange_block>>>(data, reverse, r_len);



            cudaFree(data_d);
            cudaFree(reverse_d);
            free(reverse);
        }
    };
}