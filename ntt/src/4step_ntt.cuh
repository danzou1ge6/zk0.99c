#include "ntt.cuh"
#include "inplace_transpose/cuda/transpose.cuh"
#include "recompute_ntt.cuh"
#include "self_sort_in_place_ntt.cuh"
#include "transpose/matrix_transpose.h"
#include <memory>
// #include <pybind11/embed.h>
// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// #include <pybind11/pytypes.h>

#define CPU_TRANSPOSE

namespace ntt {

    template <typename Field, u32 io_group>
    void __global__ batched_ntt_with_stride (u32 *data, u32 logn, u64 stride, u32 *roots, Field *unit, u32 offset) {
        const usize WORDS = Field::LIMBS;

        // printf("offset: %d blockx: %d blocky: %d threadx: %d thready: %d\n", offset, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
        data += blockDim.y * blockIdx.x * WORDS + threadIdx.y * WORDS; // previous ntts

        using WarpExchangeT = cub::WarpExchange<u32, io_group, io_group>;
        extern __shared__ typename WarpExchangeT::TempStorage temp_storage[];

        const u32 lid = threadIdx.x;

        Field a, b;

        // Read data
        a = Field::load(data + lid * stride * WORDS);
        b = Field::load(data + (lid * stride + stride  * (1 << (logn - 1))) * WORDS);
        // printf("a: %u b: %u\n", a.n.limbs[0], b.n.limbs[0]);


        // TODO: use exchange
        // a = mont::load_exchange<Field, io_group>(data, [&](int id)->u64{return lid * stride;}, temp_storage);
        // b = mont::load_exchange<Field, io_group>(data, [&](int id)->u64{return lid * stride + stride  * (1 << (logn - 1));}, temp_storage);

        for (u32 i = 0; i < logn; i++) {
            if (i != 0) {
                u32 lanemask = 1 << (logn - i - 1);
                Field tmp;
                tmp = ((lid / lanemask) & 1) ? a : b;

                #pragma unroll
                for (u32 j = 0; j < WORDS; j++) {
                    tmp.n.limbs[j] = __shfl_xor_sync(0xffffffff, tmp.n.limbs[j], lanemask);
                }

                if ((lid / lanemask) & 1) a = tmp;
                else b = tmp;
            }

            auto tmp = a;
            a = a + b;
            b = tmp - b;

            u32 bit = (1 << (logn - 1)) >> i;
            u64 di = (lid & (bit - 1));

            if (di != 0) {
                auto w = Field::load(roots + (di << i) * WORDS);
                b = b * w;
            }
        }

        u32 ida = __brev(lid << 1) >> (32 - logn);
        u32 idb = __brev((lid << 1) + 1) >> (32 - logn);

        // printf("%u %u\n", (offset + blockDim.y * blockIdx.x + threadIdx.y) * ida, (offset + blockDim.y * blockIdx.x + threadIdx.y) * idb);
        
        a = a * (*unit).pow((offset + blockDim.y * blockIdx.x + threadIdx.y) * ida);
        b = b * (*unit).pow((offset + blockDim.y * blockIdx.x + threadIdx.y) * idb);

        // Write back
        a.store(data + ida * stride * WORDS);
        b.store(data + idb * stride * WORDS);
        
        // mont::store_exchange<Field, io_group>(a, data, [&](u32 id)->u64{return id << 1;}, temp_storage);
        // mont::store_exchange<Field, io_group>(b, data, [&](u32 id)->u64{return (id << 1) + 1;}, temp_storage);
    }

    template <typename Field>
    Field inline get_unit(const u32 *omega, u32 logn) {
        using Number = mont::Number<Field::LIMBS>;
        auto unit = Field::from_number(Number::load(omega));
        auto one = Number::zero();
        one.limbs[0] = 1;
        Number exponent = (Field::ParamsType::m() - one).slr(logn);
        unit = unit.pow(exponent);
        return unit;
    }

    template<typename Field>
    void offchip_ntt(u32 *input, u32 *output, int logn, const u32 *omega) {
        constexpr usize WORDS = Field::LIMBS;

        // pybind11::scoped_interpreter guard{};

        // pybind11::exec(R"(
        //     import numpy as np
        //     def transpose(src, dst, a, b, c):
        //         x = np.frombuffer(src, dtype=np.uint32)
        //         x = x.reshape(a,b,c)
        //         y = np.frombuffer(dst, dtype=np.uint32)
        //         y = y.reshape(b,a,c)
        //         np.copyto(y, np.transpose(x, (1, 0, 2)))
                
        // )");


        // 获取 Python 中的 transpose 函数
        // pybind11::object transpose_func = pybind11::module::import("__main__").attr("transpose");

        usize avail, total;
        cudaMemGetInfo(&avail, &total);

        // avail = 8ll * 1024 * 1024 * 1024;
        
        u32 lgp = 4;
        u32 lgq = logn - lgp;

        while (((1ll << lgq) + std::max(1ll << lgp, 1ll << (lgq - lgp))) * WORDS * sizeof(u32) + 1100ll * sizeof(Field) > avail) {
            lgq--;
            lgp++;
            if (lgp > logn) {
                throw std::runtime_error("Not enough memory");
            }
        }
        assert(lgp <= 6); // this will cover most cases, but if you need more, you can implement a new kernel for longer col-wise NTT

        auto rest = avail - ((1 << lgq) + std::max(1 << lgp, 1 << (lgq - lgp))) * WORDS * sizeof(u32);
        bool recompute = rest < lgq * WORDS * sizeof(Field) / 2;

        u32 len_per_line = 1 << (lgq - lgp);
        auto unit0 = get_unit<Field>(omega, logn);
        auto unit1 = get_unit<Field>(omega, lgp);
        auto unit2 = get_unit<Field>(omega, lgq);

        u32 *roots, *roots_d;
        cudaHostAlloc(&roots, (1 << lgp) / 2 * sizeof(Field), cudaHostAllocDefault);
        gen_roots_cub<Field> gen;
        gen(roots, 1 << (lgp - 1), unit1);

        std::unique_ptr<ntt::best_ntt> ntt;
        if (recompute) {
            ntt = std::make_unique<ntt::recompute_ntt<Field>>(unit2.n.limbs, lgq, false);
        } 
        else {
            ntt = std::make_unique<ntt::self_sort_in_place_ntt<Field>>(unit2.n.limbs, lgq, false);
        }

        cudaStream_t stream[2];
        cudaStreamCreate(&stream[0]);
        cudaStreamCreate(&stream[1]);

        cudaMalloc(&roots_d, (1 << lgp) / 2 * sizeof(Field));
        cudaMemcpy(roots_d, roots, (1 << lgp) / 2 * sizeof(Field), cudaMemcpyHostToDevice);

        u32 *buffer_d[2];
        cudaMallocAsync(&buffer_d[0], sizeof(Field) * (1 << lgq), stream[0]);
        cudaMallocAsync(&buffer_d[1], sizeof(Field) * (1 << lgq), stream[1]);
        
        Field *unit0_d;
        cudaMalloc(&unit0_d, sizeof(Field));
        cudaMemcpy(unit0_d, &unit0, sizeof(Field), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop, stop_cal;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventCreate(&stop_cal);
        cudaEventRecord(start);

        for (int i = 0, id = 0; i < (1 << lgq); i += len_per_line, id ^= 1) {
            for (int j = 0; j < (1 << lgp); j++) {
                auto dst = buffer_d[id] + j * WORDS * len_per_line;
                auto src = input + j * WORDS * (1 << lgq) + i * WORDS;
                cudaMemcpyAsync(dst, src, sizeof(Field) * len_per_line, cudaMemcpyHostToDevice, stream[id]);
            }
            // kernel call
            dim3 block(1 << (lgp - 1));
            block.y = std::min(256 / block.x, len_per_line);
            dim3 grid(std::max(1u, len_per_line / block.y));

            const u32 io_group = 1 << log2_int(WORDS);

            using WarpExchangeT = cub::WarpExchange<u32, io_group, io_group>;
            usize shared_size = (sizeof(typename WarpExchangeT::TempStorage) * (block.x * block.y / io_group));

            batched_ntt_with_stride<Field, io_group> <<<grid, block, shared_size, stream[id]>>>(buffer_d[id], lgp, len_per_line, roots_d, unit0_d, i);
            
            for (int j = 0; j < (1 << lgp); j++) {
                auto src = buffer_d[id] + j * WORDS * len_per_line;
                auto dst = input + j * WORDS * (1 << lgq) + i * WORDS;
                cudaMemcpyAsync(dst, src, sizeof(u32) * WORDS * len_per_line, cudaMemcpyDeviceToHost, stream[id]);
            }
        }
        cudaStreamSynchronize(stream[0]);
        cudaStreamSynchronize(stream[1]);

        cudaFree(roots_d);
        cudaFree(unit0_d);

        ntt->to_gpu();

        for (u32 i = 0, id = 0; i < (1 << lgp); i++, id ^= 1) {
            auto src = input + i * WORDS * (1 << lgq);
            cudaMemcpyAsync(buffer_d[id], src, sizeof(u32) * WORDS * (1 << lgq), cudaMemcpyHostToDevice, stream[id]);
            ntt->ntt(buffer_d[id], stream[id], 1 << lgq, true);
            cudaMemcpyAsync(src, buffer_d[id], sizeof(u32) * WORDS * (1 << lgq), cudaMemcpyDeviceToHost, stream[id]);
        }

        cudaStreamSynchronize(stream[0]);
        cudaStreamSynchronize(stream[1]);

        cudaEventRecord(stop_cal);

        #ifdef CPU_TRANSPOSE


        // auto src = pybind11::memoryview::from_buffer(input, {(1 << lgp), (1 << lgq), (int)WORDS}, {sizeof(u32) * WORDS * (1 << lgq), sizeof(u32) * WORDS, sizeof(u32)});
        // auto dst = pybind11::memoryview::from_buffer(output, {(1 << lgq), (1 << lgp), (int)WORDS}, {sizeof(u32) * WORDS * (1 << lgp), sizeof(u32) * WORDS, sizeof(u32)});


        // transpose_func(src, dst, 1ll << lgp, 1ll << lgq, WORDS);
        transpose(reinterpret_cast<LargeInteger<WORDS * 4>*>(input), reinterpret_cast<LargeInteger<WORDS * 4>*>(output), 1 << lgp, 1 << lgq);
            
        #else

        for (int i = 0, id = 0; i < (1 << lgq); i += len_per_line, id ^= 1) {
            for (int j = 0; j < (1 << lgp); j++) {
                auto dst = buffer_d[id] + j * WORDS * len_per_line;
                auto src = input + j * WORDS * (1 << lgq) + i * WORDS;
                cudaMemcpyAsync(dst, src, sizeof(Field) * len_per_line, cudaMemcpyHostToDevice, stream[id]);
            }
            
            inplace::transpose(true, (Field *)buffer_d[id], 1 << lgp, len_per_line);

            auto src = buffer_d[id];
            auto dst = output + i * WORDS * (1 << lgp);
            cudaMemcpyAsync(dst, src, sizeof(u32) * WORDS * (1 << lgq), cudaMemcpyDeviceToHost, stream[id]);
        }
        cudaStreamSynchronize(stream[0]);
        cudaStreamSynchronize(stream[1]);

        #endif

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0, cal_milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cudaEventElapsedTime(&cal_milliseconds, start, stop_cal);
        printf("k = %d, time = %f ms, cal time = %f ms\n", logn, milliseconds, cal_milliseconds);

        ntt->clean_gpu();
        cudaFree(buffer_d[0]);
        cudaFree(buffer_d[1]);
    }
}