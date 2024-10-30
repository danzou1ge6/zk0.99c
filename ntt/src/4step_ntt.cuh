#include "ntt.cuh"
#include "inplace_transpose/cuda/transpose.cuh"
#include "recompute_ntt.cuh"
#include "self_sort_in_place_ntt.cuh"
#include <__clang_cuda_builtin_vars.h>
#include <memory>

namespace ntt {

    template <typename Field, u32 io_group>
    void __global__ batched_ntt_with_stride (u32 *data, u32 logn, u32 *twiddle, Field *unit, u32 offset) {
        const u32 WORDS = Field::LIMBS;
        data += offset * WORDS + blockDim.y * blockIdx.x * WORDS + threadIdx.y * WORDS;    

        constexpr int warp_threads = io_group;
        constexpr int items_per_thread = io_group;
        const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
        u32 thread_data[io_group];

        // Specialize WarpExchange for a virtual warp of a threads owning b integer items each
        using WarpExchangeT = cub::WarpExchange<u32, items_per_thread, warp_threads>;

        // Allocate shared memory for WarpExchange
        extern __shared__ typename WarpExchangeT::TempStorage temp_storage[];

        const u32 lid = threadIdx.x & (group_sz - 1);
        const u32 lsize = group_sz;
        const u32 group_id = threadIdx.x / group_sz;
        const u32 group_num = blockDim.x / group_sz;
        const u32 index = blockIdx.x * group_num + group_id;

        const u32 lgp = log_stride - deg + 1;
        const u32 end_stride = 1 << lgp; //stride of the last butterfly

        // each segment is independent
        const u32 segment_start = (index >> lgp) << (lgp + deg);
        const u32 segment_id = index & (end_stride - 1);
        
        const u32 subblock_sz = 1 << (deg - 1); // # of neighbouring butterfly in the last round

        data += ((u64)(segment_start + segment_id)) * WORDS; // use u64 to avoid overflow

        const u32 io_id = lid & (io_group - 1);
        const u32 lid_start = lid - io_id;
        const u32 cur_io_group = io_group < lsize ? io_group : lsize;

        Field a, b;

        // Read data
        if (cur_io_group == io_group) {
            for (int i = lid_start; i != lid_start + io_group; i ++) {
                if (io_id < WORDS) {
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_id << (lgp);

                    thread_data[i - lid_start] = data[(gpos) * WORDS + io_id];
                }
            }

            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);
            a = Field::load(thread_data);
            __syncwarp();

            for (int i = lid_start; i != lid_start + io_group; i ++) {
                if (io_id < WORDS) {
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_id << (lgp);

                    thread_data[i - lid_start] = data[(gpos+ (end_stride << (deg - 1))) * WORDS + io_id];
                }
            }

            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);
            b = Field::load(thread_data);
            __syncwarp();

        } else {
            u32 group_id = lid & (subblock_sz - 1);
            u64 gpos = group_id << (lgp);

            a = Field::load(data + (gpos) * WORDS);
            b = Field::load(data + (gpos + (end_stride << (deg - 1))) * WORDS);
        }

        // printf("threadid: %d, a: %d, b: %d\n", threadIdx.x, a.n.c0, b.n.c0);
        


        barrier::arrival_token token = bar.arrive(); /* this thread arrives. Arrival does not block a thread */

        const u32 pqshift = log_len - 1 - log_stride;

        for (u32 i = 0; i < deg; i++) {
            if (i != 0) {
                u32 lanemask = 1 << (deg - i - 1);
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

            u32 bit = subblock_sz >> i;
            u64 di = (lid & (bit - 1)) * end_stride + segment_id;

            if (di != 0) {
                auto w = Field::load(roots + (di << i << pqshift) * WORDS);
                b = b * w;
            }
        }
        
        bar.wait(std::move(token)); /* wait for all threads participating in the barrier to complete bar.arrive()*/

        // Write back
        if (cur_io_group == io_group) {
            a.store(thread_data);
            WarpExchangeT(temp_storage[warp_id]).BlockedToStriped(thread_data, thread_data);
            __syncwarp();

            for (int ti = lid_start; ti != lid_start + io_group; ti ++) {
                u32 group_id = ti & (subblock_sz - 1);
                u64 gpos = group_id << (lgp + 1);
                
                if (io_id < WORDS) {
                    data[(gpos) * WORDS + io_id] = thread_data[ti - lid_start];
                }
            }

            b.store(thread_data);
            WarpExchangeT(temp_storage[warp_id]).BlockedToStriped(thread_data, thread_data);

            for (int ti = lid_start; ti != lid_start + io_group; ti ++) {
                u32 group_id = ti & (subblock_sz - 1);
                u64 gpos = group_id << (lgp + 1);
                
                if (io_id < WORDS) {
                    data[(gpos + end_stride) * WORDS + io_id] = thread_data[ti - lid_start];
                }
            }

        } else {
            u32 group_id = lid & (subblock_sz - 1);
            u64 gpos = group_id << (lgp + 1);
            
            a.store(data + (gpos) * WORDS);
            b.store(data + (gpos + end_stride) * WORDS);
        }
    }

    template <typename Field>
    Field inline get_unit(u32 *omega, u32 logn) {
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
        constexpr u32 WORDS = Field::LIMBS;

        usize avail, total;
        cudaMemGetInfo(&avail, &total);
        u32 lgp = 2;
        u32 lgq = logn - lgp;

        while (((1 << lgq) + std::max(1 << lgp, 1 << (lgq - lgp))) * WORDS * sizeof(u32) + 1100 * sizeof(Field) > avail) {
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
        } else {
            ntt = std::make_unique<ntt::self_sort_in_place_ntt<Field>>(unit2.n.limbs, lgq, false);
        }

        // begin
        cudaStream_t stream[2];
        cudaStreamCreate(&stream[0]);
        cudaStreamCreate(&stream[1]);

        cudaMalloc(&roots_d, (1 << lgp) / 2 * sizeof(Field));
        cudaMemcpy(roots_d, roots, (1 << lgp) / 2 * sizeof(Field), cudaMemcpyHostToDevice);

        u32 *buffer_d[2];
        cudaMallocAsync(&buffer_d[0], sizeof(Field) * (1 << lgq), stream[0]);
        cudaMallocAsync(&buffer_d[1], sizeof(Field) * (1 << lgq), stream[1]);
        
        u32 *unit0_d;
        cudaMalloc(&unit0_d, sizeof(Field));
        cudaMemcpy(unit0_d, &unit0, sizeof(Field), cudaMemcpyHostToDevice);

        for (int i = 0, id = 0; i < (1 << lgq); i += len_per_line, id ^= 1) {
            for (int j = 0; j < (1 << lgp); j++) {
                auto dst = buffer_d[id] + j * WORDS * len_per_line;
                auto src = input + j * WORDS * (1 << lgq) + i * WORDS;
                cudaMemcpyAsync(dst, src, sizeof(u32) * WORDS * len_per_line, cudaMemcpyHostToDevice, stream[id]);
            }
            // kernel call
            dim3 block(1 << (lgp - 1));
            block.y = 256 / block.x;
            dim3 grid(((1 << (lgq -1)) - 1) / block.x + 1);
            batched_ntt_with_stride<<<grid, block, 0, stream[id]>>>(buffer_d[id], lgq, roots_d, unit0_d, i);
            
            for (int j = 0; j < (1 << lgp); j++) {
                auto src = buffer_d[id] + j * WORDS * len_per_line;
                auto dst = input + j * WORDS * (1 << lgq) + i * WORDS;
                cudaMemcpyAsync(dst, src, sizeof(u32) * WORDS * len_per_line, cudaMemcpyDeviceToHost, stream[id]);
            }
        }
        cudaStreamSynchronize(stream[0]);
        cudaStreamSynchronize(stream[1]);

        ntt->to_gpu();

        for (u32 i = 0, id = 0; i < (1 << lgp); i++, id ^= 1) {
            auto src = input + i * WORDS * (1 << lgq);
            cudaMemcpyAsync(buffer_d[id], src, sizeof(u32) * WORDS * (1 << lgq), cudaMemcpyHostToDevice, stream[id]);

            ntt->ntt(buffer_d[id], stream[id], 1 << lgq, true);

            inplace::transpose(true, (Field *)buffer_d[id], 1 << lgp, len_per_line);

            for (u32 j = 0; j < (1 << lgp); j++) {
                auto src = buffer_d[id] + j * WORDS * len_per_line;
                auto dst = output +  i * WORDS * len_per_line + j * WORDS * (1 << lgq);
                cudaMemcpyAsync(dst, src, sizeof(u32) * WORDS * len_per_line, cudaMemcpyDeviceToHost, stream[id]);
            }
        }

        cudaStreamSynchronize(stream[0]);
        cudaStreamSynchronize(stream[1]);
        ntt->clean_gpu();
        cudaFree(buffer_d[0]);
        cudaFree(buffer_d[1]);
    }
}