#include "ntt.cuh"
#include "inplace_transpose/cuda/transpose.cuh"

namespace ntt {
    template <typename Field>
    void __global__ batched_ntt_with_stride (u32 *data, u32 logn, u32 stride, u32 *twiddle, u32 * unit, u32 group_num, u32 offset) {
        const u32 WORDS = Field::LIMBS;
        data += blockIdx.x * group_num;
        
    }

    template<typename Field>
    void offchip_ntt(u32 *input, u32 *output, int logn) {
        u32 WORDS = Field::LIMBS;
        usize avail, total;
        cudaMemGetInfo(&avail, &total);
        u32 lgp = 2;
        u32 lgq = logn - lgp;
        u32 *buffer_d[3];
        cudaMalloc(&buffer_d[0], sizeof(u32) * (1 << lgq) * WORDS);
        cudaMalloc(&buffer_d[1], sizeof(u32) * (1 << lgq) * WORDS);

        cudaStream_t stream[2];
        cudaStreamCreate(&stream[0]);
        cudaStreamCreate(&stream[1]);

        u32 len_per_line = 1 << (lgq - lgp);
        for (int i = 0, id = 0; i < (1 << lgq); i += len_per_line, id ^= 1) {
            for (int j = 0; j < (1 << lgp); j++) {
                auto dst = buffer_d[id] + j * WORDS * len_per_line;
                auto src = input + j * WORDS * (1 << lgq) + i * WORDS;
                cudaMemcpyAsync(dst, src, sizeof(u32) * WORDS * len_per_line, cudaMemcpyHostToDevice, stream[id]);
            }
            // kernel call
            dim3 block(256);
            dim3 grid(((1 << (lgq -1)) - 1) / block.x + 1);
            batched_ntt_with_stride<<<grid, block, 0, stream[id]>>>();
            
            for (int j = 0; j < (1 << lgp); j++) {
                auto src = buffer_d[id] + j * WORDS * len_per_line;
                auto dst = output + j * WORDS * (1 << lgq) + i * WORDS;
                cudaMemcpyAsync(dst, src, sizeof(u32) * WORDS * len_per_line, cudaMemcpyDeviceToHost, stream[id]);
            }
        }
        cudaStreamSynchronize(stream[0]);
        cudaStreamSynchronize(stream[1]);
        cudaFree(buffer_d[0]);
        cudaFree(buffer_d[1]);
        
        cudaMalloc(&buffer_d[0], size);
        cudaMalloc(&buffer_d[1], size);
        cudaMalloc(&buffer_d[2], size);

        cudaEvent_t event[2];
        cudaEventCreate(&event[0]);
        cudaEventCreate(&event[1]);

        int buffer_id[2] = {0, 2};

        cudaEventRecord(event[1], stream[1]);

        for (u32 i = 0, id = 0; i < 1 << lgp; i++, id ^= 1) {
            cudaMemcpyAsync(buffer_d[buffer_id[id]], const void *src, size_t count, cudaMemcpyHostToDevice, stream[id]);

            ntt(buffer_d[buffer_id[id]]);

            cudaStreamWaitEvent(stream[id^1], event[id^1]);

            inplace::transpose(bool row_major, T *data, int m, int n);

            buffer_id[id] = (buffer_id[id] + 1) % 3;
            cudaEventRecord(event[id], stream[id]);

            cudaMemcpyAsync(void *dst, buffer_d[buffer_id[id]], size_t count, cudaMemcpyDeviceToHost, stream[id]);

        }

        cudaEventDestroy(event[0]);
        cudaEventDestroy(event[1]);
    }
}