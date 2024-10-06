#pragma once

#include <bit>
#include "ntt.cuh"
#include <cassert>
#include <cstdio>

namespace ntt {
    template <u32 WORDS, u32 io_group>
    __global__ void SSIP_NTT_stage1 (u32_E * x, // input data, shape: [1, len*words] data stored in row major i.e. a_0 ... a_7 b_0 ... b_7 ...
                                    const u32_E * pq, // twiddle factors for shard memory NTT
                                    u32 log_len, // input data length
                                    u32 log_stride, // the log of the stride of initial butterfly op
                                    u32 deg, // the deg of the shared memory NTT
                                    u32 max_deg, // max deg supported by pq
                                    mont256::Params* param, // params for field ops
                                    u32 group_sz, // number of threads used in a shared memory NTT
                                    u32 * roots, // twiddle factors for global NTT
                                    bool coalesced_roots) // whether to use cub to coalesce the read of roots
    {
        extern __shared__ u32_E s[];
        // column-major in shared memory
        // data patteren:
        // a0_word0 a1_word0 a2_word0 a3_word0 ... an_word0 [empty] a0_word1 a1_word1 a2_word1 a3_word1 ...
        // for coleasced read, we need to read a0_word0, a0_word1, a0_word2, ... a0_wordn in a single read
        // so thread [0,WORDS) will read a0_word0 a0_word1 a0_word2 ... 
        // then a1_word0 a1_word1 a1_word2 ... 
        // so we need the empty space is for padding to avoid bank conflict during read because n is likely to be 32k

        const u32 lid = threadIdx.x & (group_sz - 1);
        const u32 lsize = group_sz;
        const u32 group_id = threadIdx.x / group_sz;
        const u32 group_num = blockDim.x / group_sz;
        const u32 index = blockIdx.x * group_num + group_id;

        constexpr int warp_threads = io_group;
        constexpr int items_per_thread = io_group;
        const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
        u32 thread_data[io_group];

        // Specialize WarpExchange for a virtual warp of a threads owning b integer items each
        using WarpExchangeT = cub::WarpExchange<u32, items_per_thread, warp_threads>;

        // Allocate shared memory for WarpExchange
        typename WarpExchangeT::TempStorage *temp_storage = (typename WarpExchangeT::TempStorage*) (s + group_num * ((1 << deg) + 1) * WORDS);

        auto u = s + group_id * ((1 << deg) + 1) * WORDS;

        const u32 lgp = log_stride - deg + 1;
        const u32 end_stride = 1 << lgp; //stride of the last butterfly

        // each segment is independent
        const u32 segment_start = (index >> lgp) << (lgp + deg);
        const u32 segment_id = index & (end_stride - 1);
        
        const u32 subblock_sz = 1 << (deg - 1); // # of neighbouring butterfly in the last round

        x += ((u64)(segment_start + segment_id)) * WORDS; // use u64 to avoid overflow

        const u32 io_id = lid & (io_group - 1);
        const u32 lid_start = lid - io_id;
        const u32 shared_read_stride = (lsize << 1) + 1;
        const u32 cur_io_group = io_group < lsize ? io_group : lsize;
        const u32 io_per_thread = io_group / cur_io_group;

        const u32 io_group_id = lid / cur_io_group;
        int io_st, io_ed, io_stride;

        if ((io_group_id / 2) & 1) {
            io_st = lid_start;
            io_ed = lid_start + cur_io_group;
            io_stride = 1;
        } else {
            io_st = lid_start + cur_io_group - 1;
            io_ed = ((int)lid_start) - 1;
            io_stride = -1;
        }
        // Read data
        for (int i = io_st; i != io_ed; i += io_stride) {
            for (u32 j = 0; j < io_per_thread; j++) {
                u32 io = io_id + j * cur_io_group;
                if (io < WORDS) {
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_id << (lgp + 1);
                    u[(i << 1) + io * shared_read_stride] = x[gpos * WORDS + io];
                    u[(i << 1) + 1 + io * shared_read_stride] = x[(gpos + end_stride) * WORDS + io];
                    // if (blockIdx.x == 0 && threadIdx.x < 32) printf("%d ", ((i << 1) + io * shared_read_stride) % 32);
                    // if (blockIdx.x == 0 && threadIdx.x == 0) printf("\n");
                    // if (blockIdx.x == 0 && threadIdx.x < 32) printf("%d ",  ((i << 1) + 1 + io * shared_read_stride) % 32);
                    // if (blockIdx.x == 0 && threadIdx.x == 0) printf("\n");
                    // if (blockIdx.x == 0 && threadIdx.x == 0) printf("\n");
                }
            }
        }

        auto env = mont256::Env(*param);

        __syncthreads();

        const u32 pqshift = max_deg - deg;
        for(u32 rnd = 0; rnd < deg; rnd++) {
            const u32 bit = subblock_sz >> rnd;
            const u32 di = lid & (bit - 1);
            const u32 i0 = (lid << 1) - di;
            const u32 i1 = i0 + bit;

            auto a = mont256::Element::load(u + i0, shared_read_stride);
            auto b = mont256::Element::load(u + i1, shared_read_stride);
            auto tmp = a;
            a = env.add(a, b);
            b = env.sub(tmp, b);

            if (di != 0) {
                auto w = mont256::Element::load(pq + (di << rnd << pqshift) * WORDS);
                b = env.mul(b, w);
            }
            a.store(u + i0, shared_read_stride);
            b.store(u + i1, shared_read_stride);

            __syncthreads();
        }

        // Twiddle factor
        u32 k = index & (end_stride - 1);
        u64 twiddle = ((1 << log_len) >> lgp >> deg) * k;

        mont256::Element t1, t2;
        if (coalesced_roots && cur_io_group == io_group) {
            for (u32 i = 0; i < cur_io_group; i++) {
                if (io_id < WORDS) {
                    u32 pos = twiddle * ((i + lid_start) << 1);
                    thread_data[i] = roots[pos * WORDS + io_id];
                }
            }
            // Collectively exchange data into a blocked arrangement across threads
            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);

            t1 = mont256::Element::load(thread_data);
            __syncwarp();
        } else {
            t1 = mont256::Element::load(roots + (twiddle * (lid << 1)) * WORDS);
        }

        if (coalesced_roots && cur_io_group == io_group) {
            for (u32 i = 0; i < cur_io_group; i++) {
                if (io_id < WORDS) {
                    u32 pos = twiddle * (1 + ((i + lid_start) << 1));
                    thread_data[i] = roots[pos * WORDS + io_id];
                }
            }
            // Collectively exchange data into a blocked arrangement across threads
            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);

            t2 = mont256::Element::load(thread_data);
        } else {
            t2 = mont256::Element::load(roots + (twiddle * ((lid << 1) + 1)) * WORDS);
        }

        // auto t1 = mont256::Element::load(roots);
        // auto t2 = mont256::Element::load(roots + WORDS * twiddle);

        // auto twiddle = pow_lookup_constant <WORDS> ((len >> (log_stride - deg + 1) >> deg) * k, env);
        // auto t1 = env.pow(twiddle, lid << 1, deg);
        // auto t2 = env.mul(t1, twiddle);
        
        auto pos1 = __brev(lid << 1) >> (32 - deg);
        auto pos2 = __brev((lid << 1) + 1) >> (32 - deg);

        auto a = mont256::Element::load(u + pos1, shared_read_stride);
        a = env.mul(a, t1);
        a.store(u + pos1, shared_read_stride);
        
        auto b = mont256::Element::load(u + pos2, shared_read_stride);
        b = env.mul(b, t2);
        b.store(u + pos2, shared_read_stride);

        __syncthreads();

        // Write back
        for (int i = io_st; i != io_ed; i += io_stride) {
            for (u32 j = 0; j < io_per_thread; j++) {
                u32 io = io_id + j * cur_io_group;
                if (io < WORDS) {
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_id << (lgp + 1);
                    x[gpos * WORDS + io] = u[(i << 1) + io * shared_read_stride];
                    x[(gpos + end_stride) * WORDS + io] = u[(i << 1) + 1 + io * shared_read_stride];
                }
            }
        }
    }

    template <u32 WORDS, u32 io_group>
    __global__ void SSIP_NTT_stage1_warp (u32_E * x, const u32_E * pq, u32 log_len, u32 log_stride, u32 deg, u32 max_deg, mont256::Params* param, u32 group_sz, u32 * roots, bool coalesced_roots) {
        extern __shared__ u32_E s[];

        const u32 lid = threadIdx.x & (group_sz - 1);
        const u32 lsize = group_sz;
        const u32 group_id = threadIdx.x / group_sz;
        const u32 group_num = blockDim.x / group_sz;
        const u32 index = blockIdx.x * group_num + group_id;

        constexpr int warp_threads = io_group;
        constexpr int items_per_thread = io_group;
        const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
        u32 thread_data[io_group];

        // Specialize WarpExchange for a virtual warp of a threads owning b integer items each
        using WarpExchangeT = cub::WarpExchange<u32, items_per_thread, warp_threads>;

        // Allocate shared memory for WarpExchange
        typename WarpExchangeT::TempStorage *temp_storage = (typename WarpExchangeT::TempStorage*) (s + group_num * ((1 << deg) + 1) * WORDS);

        auto u = s + group_id * ((1 << deg) + 1) * WORDS;

        const u32 lgp = log_stride - deg + 1;
        const u32 end_stride = 1 << lgp; //stride of the last butterfly

        // each segment is independent
        const u32 segment_start = (index >> lgp) << (lgp + deg);
        const u32 segment_id = index & (end_stride - 1);
        
        const u32 subblock_sz = 1 << (deg - 1); // # of neighbouring butterfly in the last round

        x += ((u64)(segment_start + segment_id)) * WORDS; // use u64 to avoid overflow

        const u32 io_id = lid & (io_group - 1);
        const u32 lid_start = lid - io_id;
        const u32 shared_read_stride = (lsize << 1) + 1;
        const u32 cur_io_group = io_group < lsize ? io_group : lsize;
        const u32 io_per_thread = io_group / cur_io_group;

        const u32 io_group_id = lid / cur_io_group;
        int io_st, io_ed, io_stride;

        if ((io_group_id / 2) & 1) {
            io_st = lid_start;
            io_ed = lid_start + cur_io_group;
            io_stride = 1;
        } else {
            io_st = lid_start + cur_io_group - 1;
            io_ed = ((int)lid_start) - 1;
            io_stride = -1;
        }
        // Read data
        for (int i = io_st; i != io_ed; i += io_stride) {
            for (u32 j = 0; j < io_per_thread; j++) {
                u32 io = io_id + j * cur_io_group;
                if (io < WORDS) {
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_id << (lgp + 1);
                    u[(i << 1) + io * shared_read_stride] = x[gpos * WORDS + io];
                    u[(i << 1) + 1 + io * shared_read_stride] = x[(gpos + end_stride) * WORDS + io];
                    // if (blockIdx.x == 0 && threadIdx.x < 32) printf("%d ", ((i << 1) + io * shared_read_stride) % 32);
                    // if (blockIdx.x == 0 && threadIdx.x == 0) printf("\n");
                    // if (blockIdx.x == 0 && threadIdx.x < 32) printf("%d ",  ((i << 1) + 1 + io * shared_read_stride) % 32);
                    // if (blockIdx.x == 0 && threadIdx.x == 0) printf("\n");
                    // if (blockIdx.x == 0 && threadIdx.x == 0) printf("\n");
                }
            }
        }

        auto env = mont256::Env(*param);

        __syncthreads();

        const u32 pqshift = max_deg - deg;
        for(u32 rnd = 0; rnd < deg; rnd += 6) {
            u32 sub_deg = min(6, deg - rnd);
            u32 warp_sz = 1 << (sub_deg - 1);
            u32 warp_id = lid / warp_sz;
            
            u32 lgp = deg - rnd - sub_deg;
            u32 end_stride = 1 << lgp;

            u32 segment_start = (warp_id >> lgp) << (lgp + sub_deg);
            u32 segment_id = warp_id & (end_stride - 1);
            
            u32 laneid = lid & (warp_sz - 1);

            u32 bit = subblock_sz >> rnd;
            u32 i0 = segment_start + segment_id + laneid * end_stride;
            u32 i1 = i0 + bit;

            auto a = mont256::Element::load(u + i0, shared_read_stride);
            auto b = mont256::Element::load(u + i1, shared_read_stride);

            for (u32 i = 0; i < sub_deg; i++) {
                if (i != 0) {
                    u32 lanemask = 1 << (sub_deg - i - 1);
                    mont256::Element tmp;
                    tmp = ((lid / lanemask) & 1) ? a : b;
                    tmp.n.c0 = __shfl_xor_sync(0xffffffff, tmp.n.c0, lanemask);
                    tmp.n.c1 = __shfl_xor_sync(0xffffffff, tmp.n.c1, lanemask);
                    tmp.n.c2 = __shfl_xor_sync(0xffffffff, tmp.n.c2, lanemask);
                    tmp.n.c3 = __shfl_xor_sync(0xffffffff, tmp.n.c3, lanemask);
                    tmp.n.c4 = __shfl_xor_sync(0xffffffff, tmp.n.c4, lanemask);
                    tmp.n.c5 = __shfl_xor_sync(0xffffffff, tmp.n.c5, lanemask);
                    tmp.n.c6 = __shfl_xor_sync(0xffffffff, tmp.n.c6, lanemask);
                    tmp.n.c7 = __shfl_xor_sync(0xffffffff, tmp.n.c7, lanemask);
                    if ((lid / lanemask) & 1) a = tmp;
                    else b = tmp;
                }

                auto tmp = a;
                a = env.add(a, b);
                b = env.sub(tmp, b);
                u32 bit = (1 << sub_deg) >> (i + 1);
                u32 di = (lid & (bit - 1)) * end_stride + segment_id;

                if (di != 0) {
                    auto w = mont256::Element::load(pq + (di << (rnd + i) << pqshift) * WORDS);
                    b = env.mul(b, w);
                }
            }            

            i0 = segment_start + segment_id + laneid * 2 * end_stride;
            i1 = i0 + end_stride;
            a.store(u + i0, shared_read_stride);
            b.store(u + i1, shared_read_stride);

            __syncthreads();
        }

        // Twiddle factor
        u32 k = index & (end_stride - 1);
        u64 twiddle = ((1 << log_len) >> lgp >> deg) * k;

        mont256::Element t1, t2;
        if (coalesced_roots && cur_io_group == io_group) {
            for (u32 i = 0; i < cur_io_group; i++) {
                if (io_id < WORDS) {
                    u32 pos = twiddle * ((i + lid_start) << 1);
                    thread_data[i] = roots[pos * WORDS + io_id];
                }
            }
            // Collectively exchange data into a blocked arrangement across threads
            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);

            t1 = mont256::Element::load(thread_data);
            __syncwarp();
        } else {
            t1 = mont256::Element::load(roots + (twiddle * (lid << 1)) * WORDS);
        }

        if (coalesced_roots && cur_io_group == io_group) {
            for (u32 i = 0; i < cur_io_group; i++) {
                if (io_id < WORDS) {
                    u32 pos = twiddle * (1 + ((i + lid_start) << 1));
                    thread_data[i] = roots[pos * WORDS + io_id];
                }
            }
            // Collectively exchange data into a blocked arrangement across threads
            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);

            t2 = mont256::Element::load(thread_data);
        } else {
            t2 = mont256::Element::load(roots + (twiddle * ((lid << 1) + 1)) * WORDS);
        }

        // auto t1 = mont256::Element::load(roots);
        // auto t2 = mont256::Element::load(roots + WORDS * twiddle);

        // auto twiddle = pow_lookup_constant <WORDS> ((len >> (log_stride - deg + 1) >> deg) * k, env);
        // auto t1 = env.pow(twiddle, lid << 1, deg);
        // auto t2 = env.mul(t1, twiddle);
        
        auto pos1 = __brev(lid << 1) >> (32 - deg);
        auto pos2 = __brev((lid << 1) + 1) >> (32 - deg);

        auto a = mont256::Element::load(u + pos1, shared_read_stride);
        a = env.mul(a, t1);
        a.store(u + pos1, shared_read_stride);
        
        auto b = mont256::Element::load(u + pos2, shared_read_stride);
        b = env.mul(b, t2);
        b.store(u + pos2, shared_read_stride);

        __syncthreads();

        // Write back
        for (int i = io_st; i != io_ed; i += io_stride) {
            for (u32 j = 0; j < io_per_thread; j++) {
                u32 io = io_id + j * cur_io_group;
                if (io < WORDS) {
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_id << (lgp + 1);
                    x[gpos * WORDS + io] = u[(i << 1) + io * shared_read_stride];
                    x[(gpos + end_stride) * WORDS + io] = u[(i << 1) + 1 + io * shared_read_stride];
                }
            }
        }
    }

    template <u32 WORDS, u32 io_group>
    __global__ void SSIP_NTT_stage2_two_per_thread (u32_E * data, const u32_E * pq, u32 log_len, u32 log_stride, u32 deg, u32 max_deg, mont256::Params* param, u32 group_sz, u32 * roots, bool coalesced_roots) {
        extern __shared__ u32_E s[];
        
        u32 lid = threadIdx.x & (group_sz - 1);
        const u32 lsize = group_sz;
        const u32 group_id = threadIdx.x / group_sz;
        const u32 group_num = blockDim.x / group_sz;
        const u32 index = blockIdx.x * group_num + group_id;

        constexpr int warp_threads = io_group;
        constexpr int items_per_thread = io_group;
        const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
        u32 thread_data[io_group];

        // Specialize WarpExchange for a virtual warp of a threads owning b integer items each
        using WarpExchangeT = cub::WarpExchange<u32, items_per_thread, warp_threads>;

        // Allocate shared memory for WarpExchange
        typename WarpExchangeT::TempStorage *temp_storage = (typename WarpExchangeT::TempStorage*) (s + group_num * ((1 << (deg << 1)) + 1) * WORDS);

        u32 log_end_stride = (log_stride - deg + 1);
        u32 end_stride = 1 << log_end_stride; //stride of the last butterfly
        u32 end_pair_stride = 1 << (log_len - log_stride - 2 + deg); // the stride between the last pair of butterfly

        // each segment is independent
        // uint segment_stride = end_pair_stride << 1; // the distance between two segment
        u32 log_segment_num = (log_len - log_stride - 1 - deg); // log of # of blocks in a segment
        
        u32 segment_start = (index >> log_segment_num) << (log_segment_num + (deg << 1)); // segment_start = index / segment_num * segment_stride;

        u32 segment_id = index & ((1 << log_segment_num) - 1); // segment_id = index & (segment_num - 1);
        
        u32 subblock_sz = 1 << (deg - 1); // # of neighbouring butterfly in the last round
        u32 subblock_offset = (segment_id >> log_end_stride) << (deg + log_end_stride); // subblock_offset = (segment_id / (end_stride)) * (2 * subblock_sz * end_stride);
        u32 subblock_id = segment_id & (end_stride - 1);

        data += ((u64)(segment_start + subblock_offset + subblock_id)) * WORDS; // use u64 to avoid overflow
        auto u = s + group_id * ((1 << (deg << 1)) + 1) * WORDS;

        const u32 io_id = lid & (io_group - 1);
        const u32 lid_start = lid - io_id;
        const u32 shared_read_stride = (lsize << 1) + 1;
        const u32 cur_io_group = io_group < lsize ? io_group : lsize;
        const u32 io_per_thread = io_group / cur_io_group;

        const u32 io_group_id = lid / cur_io_group;
        int io_st, io_ed, io_stride;

        if ((io_group_id / 2) & 1) {
            io_st = lid_start;
            io_ed = lid_start + cur_io_group;
            io_stride = 1;
        } else {
            io_st = lid_start + cur_io_group - 1;
            io_ed = ((int)lid_start) - 1;
            io_stride = -1;
        }

        // Read data
        for (int ti = io_st; ti != io_ed; ti += io_stride) {
            u32 second_half = (ti >= (lsize >> 1));
            u32 i = ti - second_half * (lsize >> 1);
            for (u32 j = 0; j < io_per_thread; j++) {
                u32 io = io_id + j * cur_io_group;
                if (io < WORDS) {
                    u32 group_offset = (i >> (deg - 1)) << (log_len - log_stride - 1);
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_offset + (group_id << (log_end_stride + 1)); // group_offset + group_id * (end_stride << 1)
                    u32 offset = io * shared_read_stride;

                    u[(i << 1) + offset + second_half * lsize] = data[(gpos + second_half * end_pair_stride) * WORDS + io];
                    u[(i << 1) + 1 + offset + second_half * lsize] = data[(gpos + end_stride + second_half * end_pair_stride) * WORDS + io];
                }
            }
        }
        
        auto env = mont256::Env(*param);

        __syncthreads();

        const u32 pqshift = max_deg - deg;
        u32 second_half = (lid >= (lsize >> 1));
        lid -= second_half * (lsize >> 1);
        if (second_half) {
            u += lsize;
        }

        for(u32 rnd = 0; rnd < deg; rnd++) {
            const u32 bit = subblock_sz >> rnd;
            const u32 di = lid & (bit - 1);
            const u32 i0 = (lid << 1) - di;
            const u32 i1 = i0 + bit;

            auto a = mont256::Element::load(u + i0, shared_read_stride);
            auto b = mont256::Element::load(u + i1, shared_read_stride);
            auto tmp1 = a;

            a = env.add(a, b);
            b = env.sub(tmp1, b);

            if (di != 0) {
                auto w = mont256::Element::load(pq + (di << rnd << pqshift) * WORDS);
                b = env.mul(b, w);
            }

            a.store(u + i0, shared_read_stride);
            b.store(u + i1, shared_read_stride);

            __syncthreads();
        }
        if (second_half) {
            u -= lsize;
        }
        // if (threadIdx.x == 0) {
        //     auto tu = u;
        //     printf("group_num: %d\n", group_num);
        //     for(int j  = 0; j < group_num; j++) {
        //         for (u32 i = 0; i < (1 << (deg * 2)); i++) {
        //             auto a = mont256::Element::load(tu + i, shared_read_stride);
        //             printf("%d ", a.n.c0);
        //         }
        //         tu += ((1 << (deg << 1)) + 1) * WORDS;
        //     }
        //     printf("\n");
        // }

        // Twiddle factor
        u32 k = index & (end_stride - 1);
        u32 n = 1 << log_len;

        // auto twiddle = pow_lookup_constant<WORDS>((n >> (log_stride - deg + 1) >> deg) * k, env);
        // auto twiddle_gap = pow_lookup_constant<WORDS>((n >> (log_stride - deg + 1) >> deg) * k * (1 << (deg - 1)), env);
        // auto t1 = env.pow(twiddle, lid << 1 >> deg, deg);
        // auto t2 = env.mul(t1, twiddle_gap);//env.pow(twiddle, ((lid << 1) >> deg) + ((lsize <<1) >> deg), deg);

        u64 twiddle = (n >> log_end_stride >> deg) * k;

        mont256::Element t1;
        if (coalesced_roots && cur_io_group == io_group) {
            for (u32 ti = lid_start; ti < lid_start + cur_io_group; ti++) {
                u32 second_half = (ti >= (lsize >> 1));
                u32 i = second_half ? ti - (lsize >> 1) : ti;
                if (io_id < WORDS) {
                    u32 pos = twiddle * (i << 1 >> deg) + second_half * twiddle * (1 << (deg - 1));
                    thread_data[ti - lid_start] = roots[pos * WORDS + io_id];
                }
            }
            // Collectively exchange data into a blocked arrangement across threads
            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);

            t1 = mont256::Element::load(thread_data);
        } else {
            t1 = mont256::Element::load(roots + twiddle * WORDS * (lid << 1 >> deg) + second_half * twiddle * WORDS * (1 << (deg - 1)));
            // printf("twiddle: %lld lid: %d threadid: %d \n", twiddle * (lid << 1 >> deg) + second_half * twiddle * (1 << (deg - 1)), lid, threadIdx.x);
        }

        // roots += twiddle * WORDS * (lid << 1 >> deg);
        // auto t1 = mont256::Element::load(roots);
        // roots += twiddle * WORDS * (1 << (deg - 1));
        // auto t2 = mont256::Element::load(roots);
        
        u32 a, b;
        a = __brev((lid << 1) + second_half * (lsize)) >> (32 - (deg << 1));
        b = __brev((lid << 1) + 1 + second_half * (lsize)) >> (32 - (deg << 1));

        // printf("a: %d b: %d threadid: %d \n", a, b, threadIdx.x);
        auto num = mont256::Element::load(u + a, shared_read_stride);
        num = env.mul(num, t1);
        num.store(u + a, shared_read_stride);

        num = mont256::Element::load(u + b, shared_read_stride);
        num = env.mul(num, t1);
        num.store(u + b, shared_read_stride);

        __syncthreads();

        // if (threadIdx.x == 0) {
        //     auto tu = u;
        //     printf("group_num: %d\n", group_num);
        //     for(int j  = 0; j < group_num; j++) {
        //         for (u32 i = 0; i < (1 << (deg * 2)); i++) {
        //             auto a = mont256::Element::load(tu + i, shared_read_stride);
        //             printf("%d ", a.n.c0);
        //         }
        //         tu += ((1 << (deg << 1)) + 1) * WORDS;
        //     }
        //     printf("\n");
        // }

        // Write back
        for (int ti = io_st; ti != io_ed; ti += io_stride) {
            u32 second_half = (ti >= (lsize >> 1));
            u32 i = ti - second_half * (lsize >> 1);
            for (u32 j = 0; j < io_per_thread; j++) {
                u32 io = io_id + j * cur_io_group;
                if (io < WORDS) {
                    u32 group_offset = (i >> (deg - 1)) << (log_len - log_stride - 1);
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_offset + (group_id << (log_end_stride + 1)); // group_offset + group_id * (end_stride << 1)
                    u32 offset = io * shared_read_stride;
                    a = __brev((i << 1) + second_half * (lsize)) >> (32 - (deg << 1));
                    b = __brev((i << 1) + 1 + second_half * (lsize)) >> (32 - (deg << 1));

                    data[(gpos + second_half * end_pair_stride) * WORDS + io] = u[a + offset];
                    data[(gpos + end_stride + second_half * end_pair_stride) * WORDS + io] = u[b + offset];

                }
            }
        }
    }

    template <u32 WORDS, u32 io_group>
    __global__ void SSIP_NTT_stage2 (u32_E * data, const u32_E * pq, u32 log_len, u32 log_stride, u32 deg, u32 max_deg, mont256::Params* param, u32 group_sz, u32 * roots, bool coalesced_roots) {
        extern __shared__ u32_E s[];
        
        const u32 lid = threadIdx.x & (group_sz - 1);
        const u32 lsize = group_sz;
        const u32 group_id = threadIdx.x / group_sz;
        const u32 group_num = blockDim.x / group_sz;
        const u32 index = blockIdx.x * group_num + group_id;

        constexpr int warp_threads = io_group;
        constexpr int items_per_thread = io_group;
        const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
        u32 thread_data[io_group];

        // Specialize WarpExchange for a virtual warp of a threads owning b integer items each
        using WarpExchangeT = cub::WarpExchange<u32, items_per_thread, warp_threads>;

        // Allocate shared memory for WarpExchange
        typename WarpExchangeT::TempStorage *temp_storage = (typename WarpExchangeT::TempStorage*) (s + group_num * ((1 << (deg << 1)) + 1) * WORDS);

        u32 log_end_stride = (log_stride - deg + 1);
        u32 end_stride = 1 << log_end_stride; //stride of the last butterfly
        u32 end_pair_stride = 1 << (log_len - log_stride - 2 + deg); // the stride between the last pair of butterfly

        // each segment is independent
        // uint segment_stride = end_pair_stride << 1; // the distance between two segment
        u32 log_segment_num = (log_len - log_stride - 1 - deg); // log of # of blocks in a segment
        
        u32 segment_start = (index >> log_segment_num) << (log_segment_num + (deg << 1)); // segment_start = index / segment_num * segment_stride;

        u32 segment_id = index & ((1 << log_segment_num) - 1); // segment_id = index & (segment_num - 1);
        
        u32 subblock_sz = 1 << (deg - 1); // # of neighbouring butterfly in the last round
        u32 subblock_offset = (segment_id >> log_end_stride) << (deg + log_end_stride); // subblock_offset = (segment_id / (end_stride)) * (2 * subblock_sz * end_stride);
        u32 subblock_id = segment_id & (end_stride - 1);

        data += ((u64)(segment_start + subblock_offset + subblock_id)) * WORDS; // use u64 to avoid overflow
        auto u = s + group_id * ((1 << (deg << 1)) + 1) * WORDS;

        const u32 io_id = lid & (io_group - 1);
        const u32 lid_start = lid - io_id;
        const u32 shared_read_stride = (lsize << 2) + 1;
        const u32 cur_io_group = io_group < lsize ? io_group : lsize;
        const u32 io_per_thread = io_group / cur_io_group;

        const u32 io_group_id = lid / cur_io_group;
        int io_st, io_ed, io_stride;

        if ((io_group_id / 2) & 1) {
            io_st = lid_start;
            io_ed = lid_start + cur_io_group;
            io_stride = 1;
        } else {
            io_st = lid_start + cur_io_group - 1;
            io_ed = ((int)lid_start) - 1;
            io_stride = -1;
        }

        // Read data
        for (int i = io_st; i != io_ed; i += io_stride) {
            for (u32 j = 0; j < io_per_thread; j++) {
                u32 io = io_id + j * cur_io_group;
                if (io < WORDS) {
                    u32 group_offset = (i >> (deg - 1)) << (log_len - log_stride - 1);
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_offset + (group_id << (log_end_stride + 1)); // group_offset + group_id * (end_stride << 1)
                    u32 offset = io * shared_read_stride;

                    u[(i << 1) + offset] = data[gpos * WORDS + io];
                    u[(i << 1) + 1 + offset] = data[(gpos + end_stride) * WORDS + io];
                    u[(i << 1) + (lsize << 1) + offset] = data[(gpos + end_pair_stride) * WORDS + io];
                    u[(i << 1) + (lsize << 1) + 1 + offset] = data[(gpos + end_pair_stride + end_stride) * WORDS + io];

                }
            }
        }
        
        auto env = mont256::Env(*param);

        __syncthreads();

        const u32 pqshift = max_deg - deg;
        for(u32 rnd = 0; rnd < deg; rnd++) {
        
            const u32 bit = subblock_sz >> rnd;
            const u32 di = lid & (bit - 1);
            const u32 i0 = (lid << 1) - di;
            const u32 i1 = i0 + bit;
            const u32 i2 = i0 + (lsize << 1);
            const u32 i3 = i2 + bit;

            auto a = mont256::Element::load(u + i0, shared_read_stride);
            auto b = mont256::Element::load(u + i1, shared_read_stride);
            auto c = mont256::Element::load(u + i2, shared_read_stride);
            auto d = mont256::Element::load(u + i3, shared_read_stride);
            auto tmp1 = a;
            auto tmp2 = c;

            a = env.add(a, b);
            c = env.add(c, d);
            b = env.sub(tmp1, b);
            d = env.sub(tmp2, d);

            if (di != 0) {
                auto w = mont256::Element::load(pq + (di << rnd << pqshift) * WORDS);
                b = env.mul(b, w);
                d = env.mul(d, w);
            }

            a.store(u + i0, shared_read_stride);
            b.store(u + i1, shared_read_stride);
            c.store(u + i2, shared_read_stride);
            d.store(u + i3, shared_read_stride);

            __syncthreads();
        }

        // Twiddle factor
        u32 k = index & (end_stride - 1);
        u32 n = 1 << log_len;

        // auto twiddle = pow_lookup_constant<WORDS>((n >> (log_stride - deg + 1) >> deg) * k, env);
        // auto twiddle_gap = pow_lookup_constant<WORDS>((n >> (log_stride - deg + 1) >> deg) * k * (1 << (deg - 1)), env);
        // auto t1 = env.pow(twiddle, lid << 1 >> deg, deg);
        // auto t2 = env.mul(t1, twiddle_gap);//env.pow(twiddle, ((lid << 1) >> deg) + ((lsize <<1) >> deg), deg);

        u64 twiddle = (n >> log_end_stride >> deg) * k;

        mont256::Element t1, t2;
        if (coalesced_roots && cur_io_group == io_group) {
            for (u32 i = 0; i < cur_io_group; i++) {
                if (io_id < WORDS) {
                    u32 pos = twiddle * ((i + lid_start) << 1 >> deg);
                    thread_data[i] = roots[pos * WORDS + io_id];
                }
            }
            // Collectively exchange data into a blocked arrangement across threads
            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);

            t1 = mont256::Element::load(thread_data);
            __syncwarp();
        } else {
            t1 = mont256::Element::load(roots + twiddle * WORDS * (lid << 1 >> deg));
        }

        if (coalesced_roots && cur_io_group == io_group) {
            for (u32 i = 0; i < cur_io_group; i++) {
                if (io_id < WORDS) {
                    u32 pos = twiddle * ((i + lid_start) << 1 >> deg) + twiddle * (1 << (deg - 1));
                    thread_data[i] = roots[pos * WORDS + io_id];
                }
            }
            // Collectively exchange data into a blocked arrangement across threads
            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);

            t2 = mont256::Element::load(thread_data);
        } else {
            t2 = mont256::Element::load(roots + twiddle * WORDS * (lid << 1 >> deg) + twiddle * WORDS * (1 << (deg - 1)));
        }

        // roots += twiddle * WORDS * (lid << 1 >> deg);
        // auto t1 = mont256::Element::load(roots);
        // roots += twiddle * WORDS * (1 << (deg - 1));
        // auto t2 = mont256::Element::load(roots);
        
        u32 a, b, c, d;
        a = __brev(lid << 1) >> (32 - (deg << 1));
        b = __brev((lid << 1) + 1) >> (32 - (deg << 1));
        c = __brev((lid << 1) + (lsize << 1)) >> (32 - (deg << 1));
        d = __brev((lid << 1) + (lsize << 1) + 1) >> (32 - (deg << 1));

        auto num = mont256::Element::load(u + a, shared_read_stride);
        num = env.mul(num, t1);
        num.store(u + a, shared_read_stride);

        num = mont256::Element::load(u + b, shared_read_stride);
        num = env.mul(num, t1);
        num.store(u + b, shared_read_stride);

        num = mont256::Element::load(u + c, shared_read_stride);
        num = env.mul(num, t2);
        num.store(u + c, shared_read_stride);

        num = mont256::Element::load(u + d, shared_read_stride);
        num = env.mul(num, t2);
        num.store(u + d, shared_read_stride);

        __syncthreads();

        // Write back
        for (int i = io_st; i != io_ed; i += io_stride) {
            for (u32 j = 0; j < io_per_thread; j++) {
                u32 io = io_id + j * cur_io_group;
                if (io < WORDS) {
                    u32 group_offset = (i >> (deg - 1)) << (log_len - log_stride - 1);
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_offset + (group_id << (log_end_stride + 1)); // group_offset + group_id * (end_stride << 1)
                    u32 offset = io * shared_read_stride;
                    a = __brev(i << 1) >> (32 - (deg << 1));
                    b = __brev((i << 1) + 1) >> (32 - (deg << 1));
                    c = __brev((i << 1) + (lsize << 1)) >> (32 - (deg << 1));
                    d = __brev((i << 1) + (lsize << 1) + 1) >> (32 - (deg << 1));

                    data[gpos * WORDS + io] = u[a + offset];
                    data[(gpos + end_stride) * WORDS + io] = u[b + offset];
                    data[(gpos + end_pair_stride) * WORDS + io] = u[c + offset];
                    data[(gpos + end_pair_stride + end_stride) * WORDS + io] = u[d + offset];

                }
            }
        }
    }

    template <u32 WORDS, u32 io_group>
    __global__ void SSIP_NTT_stage2_two_per_thread_warp (u32_E * data, const u32_E * pq, u32 log_len, u32 log_stride, u32 deg, u32 max_deg, mont256::Params* param, u32 group_sz, u32 * roots, bool coalesced_roots) {
        extern __shared__ u32_E s[];

        u32 lid = threadIdx.x & (group_sz - 1);
        const u32 lsize = group_sz;
        const u32 group_id = threadIdx.x / group_sz;
        const u32 group_num = blockDim.x / group_sz;
        const u32 index = blockIdx.x * group_num + group_id;

        constexpr int warp_threads = io_group;
        constexpr int items_per_thread = io_group;
        const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
        u32 thread_data[io_group];

        // Specialize WarpExchange for a virtual warp of a threads owning b integer items each
        using WarpExchangeT = cub::WarpExchange<u32, items_per_thread, warp_threads>;

        // Allocate shared memory for WarpExchange
        typename WarpExchangeT::TempStorage *temp_storage = (typename WarpExchangeT::TempStorage*) (s + group_num * ((1 << (deg << 1)) + 1) * WORDS);

        u32 log_end_stride = (log_stride - deg + 1);
        u32 end_stride = 1 << log_end_stride; //stride of the last butterfly
        u32 end_pair_stride = 1 << (log_len - log_stride - 2 + deg); // the stride between the last pair of butterfly

        // each segment is independent
        // uint segment_stride = end_pair_stride << 1; // the distance between two segment
        u32 log_segment_num = (log_len - log_stride - 1 - deg); // log of # of blocks in a segment
        
        u32 segment_start = (index >> log_segment_num) << (log_segment_num + (deg << 1)); // segment_start = index / segment_num * segment_stride;

        u32 segment_id = index & ((1 << log_segment_num) - 1); // segment_id = index & (segment_num - 1);
        
        u32 subblock_sz = 1 << (deg - 1); // # of neighbouring butterfly in the last round
        u32 subblock_offset = (segment_id >> log_end_stride) << (deg + log_end_stride); // subblock_offset = (segment_id / (end_stride)) * (2 * subblock_sz * end_stride);
        u32 subblock_id = segment_id & (end_stride - 1);

        data += ((u64)(segment_start + subblock_offset + subblock_id)) * WORDS; // use u64 to avoid overflow
        auto u = s + group_id * ((1 << (deg << 1)) + 1) * WORDS;

        const u32 io_id = lid & (io_group - 1);
        const u32 lid_start = lid - io_id;
        const u32 shared_read_stride = (lsize << 1) + 1;
        const u32 cur_io_group = io_group < lsize ? io_group : lsize;
        const u32 io_per_thread = io_group / cur_io_group;

        const u32 io_group_id = lid / cur_io_group;
        int io_st, io_ed, io_stride;

        if ((io_group_id / 2) & 1) {
            io_st = lid_start;
            io_ed = lid_start + cur_io_group;
            io_stride = 1;
        } else {
            io_st = lid_start + cur_io_group - 1;
            io_ed = ((int)lid_start) - 1;
            io_stride = -1;
        }

        // Read data
        for (int ti = io_st; ti != io_ed; ti += io_stride) {
            u32 second_half = (ti >= (lsize >> 1));
            u32 i = ti - second_half * (lsize >> 1);
            for (u32 j = 0; j < io_per_thread; j++) {
                u32 io = io_id + j * cur_io_group;
                if (io < WORDS) {
                    u32 group_offset = (i >> (deg - 1)) << (log_len - log_stride - 1);
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_offset + (group_id << (log_end_stride + 1)); // group_offset + group_id * (end_stride << 1)
                    u32 offset = io * shared_read_stride;

                    u[(i << 1) + offset + second_half * lsize] = data[(gpos + second_half * end_pair_stride) * WORDS + io];
                    u[(i << 1) + 1 + offset + second_half * lsize] = data[(gpos + end_stride + second_half * end_pair_stride) * WORDS + io];
                }
            }
        }
        
        auto env = mont256::Env(*param);

        __syncthreads();

        const u32 pqshift = max_deg - deg;
        u32 second_half = (lid >= (lsize >> 1));
        lid -= second_half * (lsize >> 1);
        if (second_half) {
            u += lsize;
        }

        for(u32 rnd = 0; rnd < deg; rnd+=6) {
            u32 bit = subblock_sz >> rnd;
            u32 di = lid & (bit - 1);
            u32 i0 = (lid << 1) - di;
            u32 i1 = i0 + bit;

            auto a = mont256::Element::load(u + i0, shared_read_stride);
            auto b = mont256::Element::load(u + i1, shared_read_stride);

            u32 sub_deg = min(6, deg - rnd);
            for (u32 i = 0; i < sub_deg; i++) {
                if (i != 0) {
                    u32 lanemask = 1 << (sub_deg - i - 1);
                    mont256::Element tmp;
                    tmp = ((lid / lanemask) & 1) ? a : b;
                    tmp.n.c0 = __shfl_xor_sync(0xffffffff, tmp.n.c0, lanemask);
                    tmp.n.c1 = __shfl_xor_sync(0xffffffff, tmp.n.c1, lanemask);
                    tmp.n.c2 = __shfl_xor_sync(0xffffffff, tmp.n.c2, lanemask);
                    tmp.n.c3 = __shfl_xor_sync(0xffffffff, tmp.n.c3, lanemask);
                    tmp.n.c4 = __shfl_xor_sync(0xffffffff, tmp.n.c4, lanemask);
                    tmp.n.c5 = __shfl_xor_sync(0xffffffff, tmp.n.c5, lanemask);
                    tmp.n.c6 = __shfl_xor_sync(0xffffffff, tmp.n.c6, lanemask);
                    tmp.n.c7 = __shfl_xor_sync(0xffffffff, tmp.n.c7, lanemask);
                    if ((lid / lanemask) & 1) a = tmp;
                    else b = tmp;
                }

                auto tmp = a;
                a = env.add(a, b);
                b = env.sub(tmp, b);
                bit = subblock_sz >> (rnd + i);
                di = lid & (bit - 1);
                if (di != 0) {
                    auto w = mont256::Element::load(pq + (di << (rnd + i) << pqshift) * WORDS);
                    b = env.mul(b, w);
                }
            }            

            i0 = (lid << 1) - di;
            i1 = i0 + bit;
            a.store(u + i0, shared_read_stride);
            b.store(u + i1, shared_read_stride);

            __syncthreads();
        }
        if (second_half) {
            u -= lsize;
        }
        // if (threadIdx.x == 0) {
        //     auto tu = u;
        //     printf("group_num: %d\n", group_num);
        //     for(int j  = 0; j < group_num; j++) {
        //         for (u32 i = 0; i < (1 << (deg * 2)); i++) {
        //             auto a = mont256::Element::load(tu + i, shared_read_stride);
        //             printf("%d ", a.n.c0);
        //         }
        //         tu += ((1 << (deg << 1)) + 1) * WORDS;
        //     }
        //     printf("\n");
        // }

        // Twiddle factor
        u32 k = index & (end_stride - 1);
        u32 n = 1 << log_len;

        // auto twiddle = pow_lookup_constant<WORDS>((n >> (log_stride - deg + 1) >> deg) * k, env);
        // auto twiddle_gap = pow_lookup_constant<WORDS>((n >> (log_stride - deg + 1) >> deg) * k * (1 << (deg - 1)), env);
        // auto t1 = env.pow(twiddle, lid << 1 >> deg, deg);
        // auto t2 = env.mul(t1, twiddle_gap);//env.pow(twiddle, ((lid << 1) >> deg) + ((lsize <<1) >> deg), deg);

        u64 twiddle = (n >> log_end_stride >> deg) * k;

        mont256::Element t1;
        if (coalesced_roots && cur_io_group == io_group) {
            for (u32 ti = lid_start; ti < lid_start + cur_io_group; ti++) {
                u32 second_half = (ti >= (lsize >> 1));
                u32 i = second_half ? ti - (lsize >> 1) : ti;
                if (io_id < WORDS) {
                    u32 pos = twiddle * (i << 1 >> deg) + second_half * twiddle * (1 << (deg - 1));
                    thread_data[ti - lid_start] = roots[pos * WORDS + io_id];
                }
            }
            // Collectively exchange data into a blocked arrangement across threads
            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);

            t1 = mont256::Element::load(thread_data);
        } else {
            t1 = mont256::Element::load(roots + twiddle * WORDS * (lid << 1 >> deg) + second_half * twiddle * WORDS * (1 << (deg - 1)));
            // printf("twiddle: %lld lid: %d threadid: %d \n", twiddle * (lid << 1 >> deg) + second_half * twiddle * (1 << (deg - 1)), lid, threadIdx.x);
        }

        // roots += twiddle * WORDS * (lid << 1 >> deg);
        // auto t1 = mont256::Element::load(roots);
        // roots += twiddle * WORDS * (1 << (deg - 1));
        // auto t2 = mont256::Element::load(roots);
        
        u32 a, b;
        a = __brev((lid << 1) + second_half * (lsize)) >> (32 - (deg << 1));
        b = __brev((lid << 1) + 1 + second_half * (lsize)) >> (32 - (deg << 1));

        // printf("a: %d b: %d threadid: %d \n", a, b, threadIdx.x);
        auto num = mont256::Element::load(u + a, shared_read_stride);
        num = env.mul(num, t1);
        num.store(u + a, shared_read_stride);

        num = mont256::Element::load(u + b, shared_read_stride);
        num = env.mul(num, t1);
        num.store(u + b, shared_read_stride);

        __syncthreads();

        // Write back
        for (int ti = io_st; ti != io_ed; ti += io_stride) {
            u32 second_half = (ti >= (lsize >> 1));
            u32 i = ti - second_half * (lsize >> 1);
            for (u32 j = 0; j < io_per_thread; j++) {
                u32 io = io_id + j * cur_io_group;
                if (io < WORDS) {
                    u32 group_offset = (i >> (deg - 1)) << (log_len - log_stride - 1);
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_offset + (group_id << (log_end_stride + 1)); // group_offset + group_id * (end_stride << 1)
                    u32 offset = io * shared_read_stride;
                    a = __brev((i << 1) + second_half * (lsize)) >> (32 - (deg << 1));
                    b = __brev((i << 1) + 1 + second_half * (lsize)) >> (32 - (deg << 1));

                    data[(gpos + second_half * end_pair_stride) * WORDS + io] = u[a + offset];
                    data[(gpos + end_stride + second_half * end_pair_stride) * WORDS + io] = u[b + offset];

                }
            }
        }
    }

    template <u32 WORDS, u32 io_group>
    __global__ void SSIP_NTT_stage2_warp (u32_E * data, const u32_E * pq, u32 log_len, u32 log_stride, u32 deg, u32 max_deg, mont256::Params* param, u32 group_sz, u32 * roots, bool coalesced_roots) {
        extern __shared__ u32_E s[];

        const u32 lid = threadIdx.x & (group_sz - 1);
        const u32 lsize = group_sz;
        const u32 group_id = threadIdx.x / group_sz;
        const u32 group_num = blockDim.x / group_sz;
        const u32 index = blockIdx.x * group_num + group_id;

        constexpr int warp_threads = io_group;
        constexpr int items_per_thread = io_group;
        const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
        u32 thread_data[io_group];

        // Specialize WarpExchange for a virtual warp of a threads owning b integer items each
        using WarpExchangeT = cub::WarpExchange<u32, items_per_thread, warp_threads>;

        // Allocate shared memory for WarpExchange
        typename WarpExchangeT::TempStorage *temp_storage = (typename WarpExchangeT::TempStorage*) (s + group_num * ((1 << (deg << 1)) + 1) * WORDS);

        u32 log_end_stride = (log_stride - deg + 1);
        u32 end_stride = 1 << log_end_stride; //stride of the last butterfly
        u32 end_pair_stride = 1 << (log_len - log_stride - 2 + deg); // the stride between the last pair of butterfly

        // each segment is independent
        // uint segment_stride = end_pair_stride << 1; // the distance between two segment
        u32 log_segment_num = (log_len - log_stride - 1 - deg); // log of # of blocks in a segment
        
        u32 segment_start = (index >> log_segment_num) << (log_segment_num + (deg << 1)); // segment_start = index / segment_num * segment_stride;

        u32 segment_id = index & ((1 << log_segment_num) - 1); // segment_id = index & (segment_num - 1);
        
        u32 subblock_sz = 1 << (deg - 1); // # of neighbouring butterfly in the last round
        u32 subblock_offset = (segment_id >> log_end_stride) << (deg + log_end_stride); // subblock_offset = (segment_id / (end_stride)) * (2 * subblock_sz * end_stride);
        u32 subblock_id = segment_id & (end_stride - 1);

        data += ((u64)(segment_start + subblock_offset + subblock_id)) * WORDS; // use u64 to avoid overflow
        auto u = s + group_id * ((1 << (deg << 1)) + 1) * WORDS;

        const u32 io_id = lid & (io_group - 1);
        const u32 lid_start = lid - io_id;
        const u32 shared_read_stride = (lsize << 2) + 1;
        const u32 cur_io_group = io_group < lsize ? io_group : lsize;
        const u32 io_per_thread = io_group / cur_io_group;

        const u32 io_group_id = lid / cur_io_group;
        int io_st, io_ed, io_stride;

        if ((io_group_id / 2) & 1) {
            io_st = lid_start;
            io_ed = lid_start + cur_io_group;
            io_stride = 1;
        } else {
            io_st = lid_start + cur_io_group - 1;
            io_ed = ((int)lid_start) - 1;
            io_stride = -1;
        }

        // Read data
        for (int i = io_st; i != io_ed; i += io_stride) {
            for (u32 j = 0; j < io_per_thread; j++) {
                u32 io = io_id + j * cur_io_group;
                if (io < WORDS) {
                    u32 group_offset = (i >> (deg - 1)) << (log_len - log_stride - 1);
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_offset + (group_id << (log_end_stride + 1)); // group_offset + group_id * (end_stride << 1)
                    u32 offset = io * shared_read_stride;

                    u[(i << 1) + offset] = data[gpos * WORDS + io];
                    u[(i << 1) + 1 + offset] = data[(gpos + end_stride) * WORDS + io];
                    u[(i << 1) + (lsize << 1) + offset] = data[(gpos + end_pair_stride) * WORDS + io];
                    u[(i << 1) + (lsize << 1) + 1 + offset] = data[(gpos + end_pair_stride + end_stride) * WORDS + io];

                }
            }
        }
        
        auto env = mont256::Env(*param);

        __syncthreads();

        const u32 pqshift = max_deg - deg;
        for(u32 rnd = 0; rnd < deg; rnd += 6) {
            u32 bit = subblock_sz >> rnd;
            u32 di = lid & (bit - 1);
            u32 i0 = (lid << 1) - di;
            u32 i1 = i0 + bit;
            u32 i2 = i0 + (lsize << 1);
            u32 i3 = i2 + bit;

            auto a = mont256::Element::load(u + i0, shared_read_stride);
            auto b = mont256::Element::load(u + i1, shared_read_stride);
            auto c = mont256::Element::load(u + i2, shared_read_stride);
            auto d = mont256::Element::load(u + i3, shared_read_stride);

            u32 sub_deg = min(6, deg - rnd);

            for (u32 i = 0; i < sub_deg; i++) {
                if (i != 0) {
                    u32 lanemask = 1 << (sub_deg - i - 1);
                    mont256::Element tmp, tmp1;
                    tmp = ((lid / lanemask) & 1) ? a : b;
                    tmp1 = ((lid / lanemask) & 1) ? c : d;
                    tmp.n.c0 = __shfl_xor_sync(0xffffffff, tmp.n.c0, lanemask);
                    tmp.n.c1 = __shfl_xor_sync(0xffffffff, tmp.n.c1, lanemask);
                    tmp.n.c2 = __shfl_xor_sync(0xffffffff, tmp.n.c2, lanemask);
                    tmp.n.c3 = __shfl_xor_sync(0xffffffff, tmp.n.c3, lanemask);
                    tmp.n.c4 = __shfl_xor_sync(0xffffffff, tmp.n.c4, lanemask);
                    tmp.n.c5 = __shfl_xor_sync(0xffffffff, tmp.n.c5, lanemask);
                    tmp.n.c6 = __shfl_xor_sync(0xffffffff, tmp.n.c6, lanemask);
                    tmp.n.c7 = __shfl_xor_sync(0xffffffff, tmp.n.c7, lanemask);
                    

                    tmp1.n.c0 = __shfl_xor_sync(0xffffffff, tmp1.n.c0, lanemask);
                    tmp1.n.c1 = __shfl_xor_sync(0xffffffff, tmp1.n.c1, lanemask);
                    tmp1.n.c2 = __shfl_xor_sync(0xffffffff, tmp1.n.c2, lanemask);
                    tmp1.n.c3 = __shfl_xor_sync(0xffffffff, tmp1.n.c3, lanemask);
                    tmp1.n.c4 = __shfl_xor_sync(0xffffffff, tmp1.n.c4, lanemask);
                    tmp1.n.c5 = __shfl_xor_sync(0xffffffff, tmp1.n.c5, lanemask);
                    tmp1.n.c6 = __shfl_xor_sync(0xffffffff, tmp1.n.c6, lanemask);
                    tmp1.n.c7 = __shfl_xor_sync(0xffffffff, tmp1.n.c7, lanemask);

                    if ((lid / lanemask) & 1) a = tmp, c = tmp1;
                    else b = tmp, d = tmp1;
                }

                auto tmp1 = a;
                auto tmp2 = c;

                a = env.add(a, b);
                c = env.add(c, d);
                b = env.sub(tmp1, b);
                d = env.sub(tmp2, d);

                bit = subblock_sz >> (rnd + i);
                di = lid & (bit - 1);

                if (di != 0) {
                    auto w = mont256::Element::load(pq + (di << (rnd + i) << pqshift) * WORDS);
                    b = env.mul(b, w);
                    d = env.mul(d, w);
                }
            }

            i0 = (lid << 1) - di;
            i1 = i0 + bit;
            i2 = i0 + (lsize << 1);
            i3 = i2 + bit;

            a.store(u + i0, shared_read_stride);
            b.store(u + i1, shared_read_stride);
            c.store(u + i2, shared_read_stride);
            d.store(u + i3, shared_read_stride);

            __syncthreads();
        }

        // Twiddle factor
        u32 k = index & (end_stride - 1);
        u32 n = 1 << log_len;

        // auto twiddle = pow_lookup_constant<WORDS>((n >> (log_stride - deg + 1) >> deg) * k, env);
        // auto twiddle_gap = pow_lookup_constant<WORDS>((n >> (log_stride - deg + 1) >> deg) * k * (1 << (deg - 1)), env);
        // auto t1 = env.pow(twiddle, lid << 1 >> deg, deg);
        // auto t2 = env.mul(t1, twiddle_gap);//env.pow(twiddle, ((lid << 1) >> deg) + ((lsize <<1) >> deg), deg);

        u64 twiddle = (n >> log_end_stride >> deg) * k;

        mont256::Element t1, t2;
        if (coalesced_roots && cur_io_group == io_group) {
            for (u32 i = 0; i < cur_io_group; i++) {
                if (io_id < WORDS) {
                    u32 pos = twiddle * ((i + lid_start) << 1 >> deg);
                    thread_data[i] = roots[pos * WORDS + io_id];
                }
            }
            // Collectively exchange data into a blocked arrangement across threads
            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);

            t1 = mont256::Element::load(thread_data);
            __syncwarp();
        } else {
            t1 = mont256::Element::load(roots + twiddle * WORDS * (lid << 1 >> deg));
        }

        if (coalesced_roots && cur_io_group == io_group) {
            for (u32 i = 0; i < cur_io_group; i++) {
                if (io_id < WORDS) {
                    u32 pos = twiddle * ((i + lid_start) << 1 >> deg) + twiddle * (1 << (deg - 1));
                    thread_data[i] = roots[pos * WORDS + io_id];
                }
            }
            // Collectively exchange data into a blocked arrangement across threads
            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);

            t2 = mont256::Element::load(thread_data);
        } else {
            t2 = mont256::Element::load(roots + twiddle * WORDS * (lid << 1 >> deg) + twiddle * WORDS * (1 << (deg - 1)));
        }

        // roots += twiddle * WORDS * (lid << 1 >> deg);
        // auto t1 = mont256::Element::load(roots);
        // roots += twiddle * WORDS * (1 << (deg - 1));
        // auto t2 = mont256::Element::load(roots);
        
        u32 a, b, c, d;
        a = __brev(lid << 1) >> (32 - (deg << 1));
        b = __brev((lid << 1) + 1) >> (32 - (deg << 1));
        c = __brev((lid << 1) + (lsize << 1)) >> (32 - (deg << 1));
        d = __brev((lid << 1) + (lsize << 1) + 1) >> (32 - (deg << 1));

        auto num = mont256::Element::load(u + a, shared_read_stride);
        num = env.mul(num, t1);
        num.store(u + a, shared_read_stride);

        num = mont256::Element::load(u + b, shared_read_stride);
        num = env.mul(num, t1);
        num.store(u + b, shared_read_stride);

        num = mont256::Element::load(u + c, shared_read_stride);
        num = env.mul(num, t2);
        num.store(u + c, shared_read_stride);

        num = mont256::Element::load(u + d, shared_read_stride);
        num = env.mul(num, t2);
        num.store(u + d, shared_read_stride);

        __syncthreads();

        // Write back
        for (int i = io_st; i != io_ed; i += io_stride) {
            for (u32 j = 0; j < io_per_thread; j++) {
                u32 io = io_id + j * cur_io_group;
                if (io < WORDS) {
                    u32 group_offset = (i >> (deg - 1)) << (log_len - log_stride - 1);
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_offset + (group_id << (log_end_stride + 1)); // group_offset + group_id * (end_stride << 1)
                    u32 offset = io * shared_read_stride;
                    a = __brev(i << 1) >> (32 - (deg << 1));
                    b = __brev((i << 1) + 1) >> (32 - (deg << 1));
                    c = __brev((i << 1) + (lsize << 1)) >> (32 - (deg << 1));
                    d = __brev((i << 1) + (lsize << 1) + 1) >> (32 - (deg << 1));

                    data[gpos * WORDS + io] = u[a + offset];
                    data[(gpos + end_stride) * WORDS + io] = u[b + offset];
                    data[(gpos + end_pair_stride) * WORDS + io] = u[c + offset];
                    data[(gpos + end_pair_stride + end_stride) * WORDS + io] = u[d + offset];

                }
            }
        }
    }

    template <u32 WORDS, u32 io_group>
    __launch_bounds__(1024) __global__ void SSIP_NTT_stage2_two_per_thread_warp_no_share (u32_E * data, const u32_E * pq, u32 log_len, u32 log_stride, u32 deg, u32 max_deg, mont256::Params* param, u32 group_sz, u32 * roots, bool coalesced_roots) {
        using barrier = cuda::barrier<cuda::thread_scope_block>;
        #pragma nv_diag_suppress static_var_with_dynamic_init
        __shared__  barrier bar;

        if (threadIdx.x == 0) {
            init(&bar, blockDim.x); // Initialize the barrier with expected arrival count
        }
        __syncthreads();

        constexpr int warp_threads = io_group;
        constexpr int items_per_thread = io_group;
        const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
        u32 thread_data[io_group];

        // Specialize WarpExchange for a virtual warp of a threads owning b integer items each
        using WarpExchangeT = cub::WarpExchange<u32, items_per_thread, warp_threads>;

        // Allocate shared memory for WarpExchange
        extern __shared__ typename WarpExchangeT::TempStorage temp_storage[];

        u32 lid = threadIdx.x & (group_sz - 1);
        const u32 lsize = group_sz;
        const u32 group_id = threadIdx.x / group_sz;
        const u32 group_num = blockDim.x / group_sz;
        const u32 index = blockIdx.x * group_num + group_id;

        u32 log_end_stride = (log_stride - deg + 1);
        u32 end_stride = 1 << log_end_stride; //stride of the last butterfly
        u32 end_pair_stride = 1 << (log_len - log_stride - 2 + deg); // the stride between the last pair of butterfly

        // each segment is independent
        // uint segment_stride = end_pair_stride << 1; // the distance between two segment
        u32 log_segment_num = (log_len - log_stride - 1 - deg); // log of # of blocks in a segment
        
        u32 segment_start = (index >> log_segment_num) << (log_segment_num + (deg << 1)); // segment_start = index / segment_num * segment_stride;

        u32 segment_id = index & ((1 << log_segment_num) - 1); // segment_id = index & (segment_num - 1);
        
        u32 subblock_sz = 1 << (deg - 1); // # of neighbouring butterfly in the last round
        u32 subblock_offset = (segment_id >> log_end_stride) << (deg + log_end_stride); // subblock_offset = (segment_id / (end_stride)) * (2 * subblock_sz * end_stride);
        u32 subblock_id = segment_id & (end_stride - 1);

        data += ((u64)(segment_start + subblock_offset + subblock_id)) * WORDS; // use u64 to avoid overflow

        const u32 io_id = lid & (io_group - 1);
        const u32 lid_start = lid - io_id;
        const u32 cur_io_group = io_group < lsize ? io_group : lsize;

        mont256::Element a, b;

        u32 second_half = (lid >= (lsize >> 1));
        lid -= second_half * (lsize >> 1);

        // Read data
        if (cur_io_group == io_group) {
            for (int ti = lid_start; ti != lid_start + io_group; ti ++) {
                u32 second_half = (ti >= (lsize >> 1));
                u32 i = ti - second_half * (lsize >> 1);
                if (io_id < WORDS) {
                    u32 group_offset = (i >> (deg - 1)) << (log_len - log_stride - 1);
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_offset + (group_id << (log_end_stride));

                    thread_data[ti - lid_start] = data[(gpos + second_half * end_pair_stride) * WORDS + io_id];
                }
            }

            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);
            a = mont256::Element::load(thread_data);
            __syncwarp();

            for (int ti = lid_start; ti != lid_start + io_group; ti ++) {
                u32 second_half = (ti >= (lsize >> 1));
                u32 i = ti - second_half * (lsize >> 1);
                if (io_id < WORDS) {
                    u32 group_offset = (i >> (deg - 1)) << (log_len - log_stride - 1);
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_offset + (group_id << (log_end_stride));

                    thread_data[ti - lid_start] = data[(gpos + (end_stride << (deg - 1)) + second_half * end_pair_stride) * WORDS + io_id];
                }
            }

            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);
            b = mont256::Element::load(thread_data);
            __syncwarp();
        } else {
            u32 group_offset = (lid >> (deg - 1)) << (log_len - log_stride - 1);
            u32 group_id = lid & (subblock_sz - 1);
            u64 gpos = group_offset + (group_id << (log_end_stride));

            a = mont256::Element::load(data + (gpos + second_half * end_pair_stride) * WORDS);
            b = mont256::Element::load(data + (gpos + (end_stride << (deg - 1)) + second_half * end_pair_stride) * WORDS);

        }

        barrier::arrival_token token = bar.arrive(); /* this thread arrives. Arrival does not block a thread */

        auto env = mont256::Env(*param);

        const u32 pqshift = max_deg - deg;

        u32 bit = subblock_sz;
        u32 di = lid & (bit - 1);

        for (u32 i = 0; i < deg; i++) {
            if (i != 0) {
                u32 lanemask = 1 << (deg - i - 1);
                mont256::Element tmp;
                tmp = ((lid / lanemask) & 1) ? a : b;
                tmp.n.c0 = __shfl_xor_sync(0xffffffff, tmp.n.c0, lanemask);
                tmp.n.c1 = __shfl_xor_sync(0xffffffff, tmp.n.c1, lanemask);
                tmp.n.c2 = __shfl_xor_sync(0xffffffff, tmp.n.c2, lanemask);
                tmp.n.c3 = __shfl_xor_sync(0xffffffff, tmp.n.c3, lanemask);
                tmp.n.c4 = __shfl_xor_sync(0xffffffff, tmp.n.c4, lanemask);
                tmp.n.c5 = __shfl_xor_sync(0xffffffff, tmp.n.c5, lanemask);
                tmp.n.c6 = __shfl_xor_sync(0xffffffff, tmp.n.c6, lanemask);
                tmp.n.c7 = __shfl_xor_sync(0xffffffff, tmp.n.c7, lanemask);
                if ((lid / lanemask) & 1) a = tmp;
                else b = tmp;
            }

            auto tmp = a;
            a = env.add(a, b);
            b = env.sub(tmp, b);
            bit = subblock_sz >> i;
            di = lid & (bit - 1);
            if (di != 0) {
                auto w = mont256::Element::load(pq + (di << i << pqshift) * WORDS);
                b = env.mul(b, w);
            }
        }

        // Twiddle factor
        u32 k = index & (end_stride - 1);
        u32 n = 1 << log_len;

        // auto twiddle = pow_lookup_constant<WORDS>((n >> (log_stride - deg + 1) >> deg) * k, env);
        // auto twiddle_gap = pow_lookup_constant<WORDS>((n >> (log_stride - deg + 1) >> deg) * k * (1 << (deg - 1)), env);
        // auto t1 = env.pow(twiddle, lid << 1 >> deg, deg);
        // auto t2 = env.mul(t1, twiddle_gap);//env.pow(twiddle, ((lid << 1) >> deg) + ((lsize <<1) >> deg), deg);

        u64 twiddle = (n >> log_end_stride >> deg) * k;

        mont256::Element t1;
        if (coalesced_roots && cur_io_group == io_group) {
            for (int i = lid_start; i < lid_start + cur_io_group; i++) {
                if (io_id < WORDS) {
                    u32 r = __brev((i << 1) + second_half * lsize) >> (32 - (deg << 1));
                    u64 pos = twiddle * (r >> deg);
                    thread_data[i - lid_start] = roots[pos * WORDS + io_id];
                }
            }

            // Collectively exchange data into a blocked arrangement across threads
            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);
            t1 = mont256::Element::load(thread_data);
            __syncwarp();
        } else {
            u32 r1 = __brev((lid << 1) + second_half * lsize) >> (32 - (deg << 1));

            u64 pos = twiddle * (r1 >> deg);
            t1 = mont256::Element::load(roots + pos * WORDS);
        }

        a = env.mul(a, t1);

        if (coalesced_roots && cur_io_group == io_group) {
           for (int i = lid_start; i < lid_start + cur_io_group; i++) {
                if (io_id < WORDS) {
                    u32 r = __brev((i << 1) + 1 + second_half * lsize) >> (32 - (deg << 1));
                    u64 pos = twiddle * (r >> deg);
                    thread_data[i - lid_start] = roots[pos * WORDS + io_id];
                }
            }

            // Collectively exchange data into a blocked arrangement across threads
            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);
            t1 = mont256::Element::load(thread_data);
            __syncwarp();
        } else {
            u32 r2 = __brev((lid << 1) + 1 + second_half * lsize) >> (32 - (deg << 1));

            u64 pos = twiddle * (r2 >> deg);
            t1 = mont256::Element::load(roots + pos * WORDS);
        }

        b = env.mul(b, t1);

        bar.wait(std::move(token)); /* wait for all threads participating in the barrier to complete bar.arrive()*/

        // Write back
        if (cur_io_group == io_group) {
            a.store(thread_data);
            WarpExchangeT(temp_storage[warp_id]).BlockedToStriped(thread_data, thread_data);
            __syncwarp();

            for (int ti = lid_start; ti != lid_start + io_group; ti ++) {
                u32 second_half = (ti >= (lsize >> 1));
                u32 i = ti - second_half * (lsize >> 1);

                u32 p;
                u32 second_half_l, gap;
                u32 lid_l;
                u32 group_offset, group_id;
                u64 gpos;

                p = __brev((i << 1) + second_half * (lsize)) >> (32 - (deg << 1));
                second_half_l = (p >= lsize);

                lid_l = (p - second_half_l * lsize);
                gap = lid_l & 1;
                lid_l = lid_l >> 1;
                

                group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
                group_id = lid_l & (subblock_sz - 1);
                gpos = group_offset + (group_id << (log_end_stride + 1));
                
                if (io_id < WORDS) {
                    data[(gpos + second_half_l * end_pair_stride + gap * end_stride) * WORDS + io_id] = thread_data[ti - lid_start];
                }
            }

            b.store(thread_data);
            WarpExchangeT(temp_storage[warp_id]).BlockedToStriped(thread_data, thread_data);

            for (int ti = lid_start; ti != lid_start + io_group; ti ++) {
                u32 second_half = (ti >= (lsize >> 1));
                u32 i = ti - second_half * (lsize >> 1);

                u32 p;
                u32 second_half_l, gap;
                u32 lid_l;
                u32 group_offset, group_id;
                u64 gpos;

                p = __brev((i << 1) + 1 + second_half * (lsize)) >> (32 - (deg << 1));
                second_half_l = (p >= lsize);

                lid_l = (p - second_half_l * lsize);
                gap = lid_l & 1;
                lid_l = lid_l >> 1;
                

                group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
                group_id = lid_l & (subblock_sz - 1);
                gpos = group_offset + (group_id << (log_end_stride + 1));
                
                if (io_id < WORDS) {
                    data[(gpos + second_half_l * end_pair_stride + gap * end_stride) * WORDS + io_id] = thread_data[ti - lid_start];
                }
            }

        } else {
            
            u32 p;
            u32 second_half_l, gap;
            u32 lid_l;
            u32 group_offset, group_id;
            u64 gpos;

            p = __brev((lid << 1) + second_half * (lsize)) >> (32 - (deg << 1));
            second_half_l = (p >= lsize);

            lid_l = (p - second_half_l * lsize);
            gap = lid_l & 1;
            lid_l = lid_l >> 1;
            

            group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
            group_id = lid_l & (subblock_sz - 1);
            gpos = group_offset + (group_id << (log_end_stride + 1));
            
            a.store(data + (gpos + second_half_l * end_pair_stride + gap * end_stride) * WORDS);

            p = __brev((lid << 1) + 1 + second_half * (lsize)) >> (32 - (deg << 1));
            second_half_l = (p >= lsize);

            lid_l = (p - second_half_l * lsize);
            gap = lid_l & 1;
            lid_l = lid_l >> 1;

            group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
            group_id = lid_l & (subblock_sz - 1);
            gpos = group_offset + (group_id << (log_end_stride + 1));
            
            b.store(data + (gpos + second_half_l * end_pair_stride + gap * end_stride) * WORDS);

        }
    }

    template <u32 WORDS, u32 io_group>
    __global__ void SSIP_NTT_stage2_warp_no_share (u32_E * data, const u32_E * pq, u32 log_len, u32 log_stride, u32 deg, u32 max_deg, mont256::Params* param, u32 group_sz, u32 * roots, bool coalesced_roots) {        
        using barrier = cuda::barrier<cuda::thread_scope_block>;
        #pragma nv_diag_suppress static_var_with_dynamic_init
        __shared__  barrier bar;

        if (threadIdx.x == 0) {
            init(&bar, blockDim.x); // Initialize the barrier with expected arrival count
        }
        __syncthreads();

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

        u32 log_end_stride = (log_stride - deg + 1);
        u32 end_stride = 1 << log_end_stride; //stride of the last butterfly
        u32 end_pair_stride = 1 << (log_len - log_stride - 2 + deg); // the stride between the last pair of butterfly

        // each segment is independent
        // uint segment_stride = end_pair_stride << 1; // the distance between two segment
        u32 log_segment_num = (log_len - log_stride - 1 - deg); // log of # of blocks in a segment
        
        u32 segment_start = (index >> log_segment_num) << (log_segment_num + (deg << 1)); // segment_start = index / segment_num * segment_stride;

        u32 segment_id = index & ((1 << log_segment_num) - 1); // segment_id = index & (segment_num - 1);
        
        u32 subblock_sz = 1 << (deg - 1); // # of neighbouring butterfly in the last round
        u32 subblock_offset = (segment_id >> log_end_stride) << (deg + log_end_stride); // subblock_offset = (segment_id / (end_stride)) * (2 * subblock_sz * end_stride);
        u32 subblock_id = segment_id & (end_stride - 1);

        data += ((u64)(segment_start + subblock_offset + subblock_id)) * WORDS; // use u64 to avoid overflow

        const u32 io_id = lid & (io_group - 1);
        const u32 lid_start = lid - io_id;
        const u32 cur_io_group = io_group < lsize ? io_group : lsize;

        mont256::Element a, b, c, d;

        // Read data
        if (cur_io_group == io_group) {
            for (int i = lid_start; i != lid_start + io_group; i ++) {
                if (io_id < WORDS) {
                    u32 group_offset = (i >> (deg - 1)) << (log_len - log_stride - 1);
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_offset + (group_id << (log_end_stride));

                    thread_data[i - lid_start] = data[(gpos) * WORDS + io_id];
                }
            }

            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);
            a = mont256::Element::load(thread_data);
            __syncwarp();

            for (int i = lid_start; i != lid_start + io_group; i ++) {
                if (io_id < WORDS) {
                    u32 group_offset = (i >> (deg - 1)) << (log_len - log_stride - 1);
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_offset + (group_id << (log_end_stride));

                    thread_data[i - lid_start] = data[(gpos+ (end_stride << (deg - 1))) * WORDS + io_id];
                }
            }

            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);
            b = mont256::Element::load(thread_data);
            __syncwarp();

            for (int i = lid_start; i != lid_start + io_group; i ++) {
                if (io_id < WORDS) {
                    u32 group_offset = (i >> (deg - 1)) << (log_len - log_stride - 1);
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_offset + (group_id << (log_end_stride));

                    thread_data[i - lid_start] = data[(gpos+ end_pair_stride) * WORDS + io_id];
                }
            }

            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);
            c = mont256::Element::load(thread_data);
            __syncwarp();

            for (int i = lid_start; i != lid_start + io_group; i ++) {
                if (io_id < WORDS) {
                    u32 group_offset = (i >> (deg - 1)) << (log_len - log_stride - 1);
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_offset + (group_id << (log_end_stride));

                    thread_data[i - lid_start] = data[(gpos + (end_stride << (deg - 1)) + end_pair_stride) * WORDS + io_id];
                }
            }

            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);
            d = mont256::Element::load(thread_data);
            __syncwarp();
        } else {
            u32 group_offset = (lid >> (deg - 1)) << (log_len - log_stride - 1);
            u32 group_id = lid & (subblock_sz - 1);
            u64 gpos = group_offset + (group_id << (log_end_stride));

            a = mont256::Element::load(data + (gpos) * WORDS);
            b = mont256::Element::load(data + (gpos + (end_stride << (deg - 1))) * WORDS);
            c = mont256::Element::load(data + (gpos + end_pair_stride) * WORDS);
            d = mont256::Element::load(data + (gpos + (end_stride << (deg - 1)) + end_pair_stride) * WORDS);
        }
        
        auto env = mont256::Env(*param);

        barrier::arrival_token token = bar.arrive(); /* this thread arrives. Arrival does not block a thread */

        const u32 pqshift = max_deg - deg;

        for (u32 i = 0; i < deg; i++) {
            if (i != 0) {
                u32 lanemask = 1 << (deg - i - 1);
                mont256::Element tmp, tmp1;
                tmp = ((lid / lanemask) & 1) ? a : b;
                tmp1 = ((lid / lanemask) & 1) ? c : d;
                tmp.n.c0 = __shfl_xor_sync(0xffffffff, tmp.n.c0, lanemask);
                tmp.n.c1 = __shfl_xor_sync(0xffffffff, tmp.n.c1, lanemask);
                tmp.n.c2 = __shfl_xor_sync(0xffffffff, tmp.n.c2, lanemask);
                tmp.n.c3 = __shfl_xor_sync(0xffffffff, tmp.n.c3, lanemask);
                tmp.n.c4 = __shfl_xor_sync(0xffffffff, tmp.n.c4, lanemask);
                tmp.n.c5 = __shfl_xor_sync(0xffffffff, tmp.n.c5, lanemask);
                tmp.n.c6 = __shfl_xor_sync(0xffffffff, tmp.n.c6, lanemask);
                tmp.n.c7 = __shfl_xor_sync(0xffffffff, tmp.n.c7, lanemask);

                tmp1.n.c0 = __shfl_xor_sync(0xffffffff, tmp1.n.c0, lanemask);
                tmp1.n.c1 = __shfl_xor_sync(0xffffffff, tmp1.n.c1, lanemask);
                tmp1.n.c2 = __shfl_xor_sync(0xffffffff, tmp1.n.c2, lanemask);
                tmp1.n.c3 = __shfl_xor_sync(0xffffffff, tmp1.n.c3, lanemask);
                tmp1.n.c4 = __shfl_xor_sync(0xffffffff, tmp1.n.c4, lanemask);
                tmp1.n.c5 = __shfl_xor_sync(0xffffffff, tmp1.n.c5, lanemask);
                tmp1.n.c6 = __shfl_xor_sync(0xffffffff, tmp1.n.c6, lanemask);
                tmp1.n.c7 = __shfl_xor_sync(0xffffffff, tmp1.n.c7, lanemask);

                if ((lid / lanemask) & 1) a = tmp, c = tmp1;
                else b = tmp, d = tmp1;
            }

            auto tmp1 = a;
            auto tmp2 = c;

            a = env.add(a, b);
            c = env.add(c, d);
            b = env.sub(tmp1, b);
            d = env.sub(tmp2, d);

            u32 bit = subblock_sz >> i;
            u32 di = lid & (bit - 1);

            if (di != 0) {
                auto w = mont256::Element::load(pq + (di << i << pqshift) * WORDS);
                b = env.mul(b, w);
                d = env.mul(d, w);
            }
        }

        // Twiddle factor
        u32 k = index & (end_stride - 1);
        u32 n = 1 << log_len;
        u64 twiddle = (n >> log_end_stride >> deg) * k;

        mont256::Element t1;
        if (coalesced_roots && cur_io_group == io_group) {
            for (int i = lid_start; i < lid_start + cur_io_group; i++) {
                if (io_id < WORDS) {
                    u32 r = __brev((i << 1)) >> (32 - (deg << 1));
                    u64 pos = twiddle * (r >> deg);
                    thread_data[i - lid_start] = roots[pos * WORDS + io_id];
                }
            }

            // Collectively exchange data into a blocked arrangement across threads
            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);
            t1 = mont256::Element::load(thread_data);
            __syncwarp();
        } else {
            u32 r1 = __brev((lid << 1)) >> (32 - (deg << 1));

            u64 pos = twiddle * (r1 >> deg);
            t1 = mont256::Element::load(roots + pos * WORDS);
        }

        a = env.mul(a, t1);

        if (coalesced_roots && cur_io_group == io_group) {
           for (int i = lid_start; i < lid_start + cur_io_group; i++) {
                if (io_id < WORDS) {
                    u32 r = __brev((i << 1) + 1) >> (32 - (deg << 1));
                    u64 pos = twiddle * (r >> deg);
                    thread_data[i - lid_start] = roots[pos * WORDS + io_id];
                }
            }

            // Collectively exchange data into a blocked arrangement across threads
            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);
            t1 = mont256::Element::load(thread_data);
            __syncwarp();
        } else {
            u32 r2 = __brev((lid << 1) + 1) >> (32 - (deg << 1));

            u64 pos = twiddle * (r2 >> deg);
            t1 = mont256::Element::load(roots + pos * WORDS);
        }

        b = env.mul(b, t1);

        if (coalesced_roots && cur_io_group == io_group) {
           for (int i = lid_start; i < lid_start + cur_io_group; i++) {
                if (io_id < WORDS) {
                    u32 r = __brev((i << 1) + lsize * 2) >> (32 - (deg << 1));
                    u64 pos = twiddle * (r >> deg);
                    thread_data[i - lid_start] = roots[pos * WORDS + io_id];
                }
            }

            // Collectively exchange data into a blocked arrangement across threads
            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);
            t1 = mont256::Element::load(thread_data);
            __syncwarp();
        } else {
            u32 r3 = __brev((lid << 1) + (lsize * 2)) >> (32 - (deg << 1));

            u64 pos = twiddle * (r3 >> deg);
            t1 = mont256::Element::load(roots + pos * WORDS);
        }

        c = env.mul(c, t1);

        if (coalesced_roots && cur_io_group == io_group) {
           for (int i = lid_start; i < lid_start + cur_io_group; i++) {
                if (io_id < WORDS) {
                    u32 r = __brev((i << 1) + 1 + lsize * 2) >> (32 - (deg << 1));
                    u64 pos = twiddle * (r >> deg);
                    thread_data[i - lid_start] = roots[pos * WORDS + io_id];
                }
            }

            // Collectively exchange data into a blocked arrangement across threads
            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);
            t1 = mont256::Element::load(thread_data);
            __syncwarp();
        } else {
            u32 r4 = __brev((lid << 1) + 1 + (lsize * 2)) >> (32 - (deg << 1));

            u64 pos = twiddle * (r4 >> deg);
            t1 = mont256::Element::load(roots + pos * WORDS);
        }

        d = env.mul(d, t1);

        // roots += twiddle * WORDS * (lid << 1 >> deg);
        // auto t1 = mont256::Element::load(roots);
        // roots += twiddle * WORDS * (1 << (deg - 1));
        // auto t2 = mont256::Element::load(roots);
        
        bar.wait(std::move(token)); /* wait for all threads participating in the barrier to complete bar.arrive()*/

        // Write back
        if (cur_io_group == io_group) {
            a.store(thread_data);
            WarpExchangeT(temp_storage[warp_id]).BlockedToStriped(thread_data, thread_data);
            __syncwarp();

            for (int ti = lid_start; ti != lid_start + io_group; ti ++) {
                u32 p;
                u32 second_half_l, gap;
                u32 lid_l;
                u32 group_offset, group_id;
                u64 gpos;

                p = __brev((ti << 1)) >> (32 - (deg << 1));
                second_half_l = (p >= lsize * 2);

                lid_l = (p - second_half_l * lsize * 2);
                gap = lid_l & 1;
                lid_l = lid_l >> 1;
                

                group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
                group_id = lid_l & (subblock_sz - 1);
                gpos = group_offset + (group_id << (log_end_stride + 1));
                
                if (io_id < WORDS) {
                    data[(gpos + second_half_l * end_pair_stride + gap * end_stride) * WORDS + io_id] = thread_data[ti - lid_start];
                }
            }

            b.store(thread_data);
            WarpExchangeT(temp_storage[warp_id]).BlockedToStriped(thread_data, thread_data);
            __syncwarp();

            for (int ti = lid_start; ti != lid_start + io_group; ti ++) {
                u32 p;
                u32 second_half_l, gap;
                u32 lid_l;
                u32 group_offset, group_id;
                u64 gpos;

                p = __brev((ti << 1) + 1) >> (32 - (deg << 1));
                second_half_l = (p >= lsize * 2);

                lid_l = (p - second_half_l * lsize * 2);
                gap = lid_l & 1;
                lid_l = lid_l >> 1;
                

                group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
                group_id = lid_l & (subblock_sz - 1);
                gpos = group_offset + (group_id << (log_end_stride + 1));
                
                if (io_id < WORDS) {
                    data[(gpos + second_half_l * end_pair_stride + gap * end_stride) * WORDS + io_id] = thread_data[ti - lid_start];
                }
            }

            c.store(thread_data);
            WarpExchangeT(temp_storage[warp_id]).BlockedToStriped(thread_data, thread_data);
            __syncwarp();

            for (int ti = lid_start; ti != lid_start + io_group; ti ++) {
                u32 p;
                u32 second_half_l, gap;
                u32 lid_l;
                u32 group_offset, group_id;
                u64 gpos;

                p = __brev((ti << 1) + lsize * 2) >> (32 - (deg << 1));
                second_half_l = (p >= lsize * 2);

                lid_l = (p - second_half_l * lsize * 2);
                gap = lid_l & 1;
                lid_l = lid_l >> 1;
                

                group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
                group_id = lid_l & (subblock_sz - 1);
                gpos = group_offset + (group_id << (log_end_stride + 1));
                
                if (io_id < WORDS) {
                    data[(gpos + second_half_l * end_pair_stride + gap * end_stride) * WORDS + io_id] = thread_data[ti - lid_start];
                }
            }

            d.store(thread_data);
            WarpExchangeT(temp_storage[warp_id]).BlockedToStriped(thread_data, thread_data);

            for (int ti = lid_start; ti != lid_start + io_group; ti ++) {
                u32 p;
                u32 second_half_l, gap;
                u32 lid_l;
                u32 group_offset, group_id;
                u64 gpos;

                p = __brev((ti << 1) + 1 + lsize * 2) >> (32 - (deg << 1));
                second_half_l = (p >= lsize * 2);

                lid_l = (p - second_half_l * lsize * 2);
                gap = lid_l & 1;
                lid_l = lid_l >> 1;
                

                group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
                group_id = lid_l & (subblock_sz - 1);
                gpos = group_offset + (group_id << (log_end_stride + 1));
                
                if (io_id < WORDS) {
                    data[(gpos + second_half_l * end_pair_stride + gap * end_stride) * WORDS + io_id] = thread_data[ti - lid_start];
                }
            }

        } else {
            
            u32 p;
            u32 second_half_l, gap;
            u32 lid_l;
            u32 group_offset, group_id;
            u64 gpos;

            p = __brev((lid << 1)) >> (32 - (deg << 1));
            second_half_l = (p >= lsize * 2);

            lid_l = (p - second_half_l * lsize * 2);
            gap = lid_l & 1;
            lid_l = lid_l >> 1;
            

            group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
            group_id = lid_l & (subblock_sz - 1);
            gpos = group_offset + (group_id << (log_end_stride + 1));
            
            a.store(data + (gpos + second_half_l * end_pair_stride + gap * end_stride) * WORDS);

            p = __brev((lid << 1) + 1) >> (32 - (deg << 1));
            second_half_l = (p >= lsize * 2);

            lid_l = (p - second_half_l * lsize * 2);
            gap = lid_l & 1;
            lid_l = lid_l >> 1;
            

            group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
            group_id = lid_l & (subblock_sz - 1);
            gpos = group_offset + (group_id << (log_end_stride + 1));
            
            b.store(data + (gpos + second_half_l * end_pair_stride + gap * end_stride) * WORDS);

            p = __brev((lid << 1) + lsize * 2) >> (32 - (deg << 1));
            second_half_l = (p >= lsize * 2);

            lid_l = (p - second_half_l * lsize * 2);
            gap = lid_l & 1;
            lid_l = lid_l >> 1;
            

            group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
            group_id = lid_l & (subblock_sz - 1);
            gpos = group_offset + (group_id << (log_end_stride + 1));
            
            c.store(data + (gpos + second_half_l * end_pair_stride + gap * end_stride) * WORDS);

            p = __brev((lid << 1) + lsize * 2 + 1) >> (32 - (deg << 1));
            second_half_l = (p >= lsize * 2);

            lid_l = (p - second_half_l * lsize * 2);
            gap = lid_l & 1;
            lid_l = lid_l >> 1;
            

            group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
            group_id = lid_l & (subblock_sz - 1);
            gpos = group_offset + (group_id << (log_end_stride + 1));
            
            d.store(data + (gpos + second_half_l * end_pair_stride + gap * end_stride) * WORDS);
        }
    }    

    template <u32 WORDS, u32 io_group, bool inverse, bool process>
    __global__ void SSIP_NTT_stage2_warp_no_share_no_twiddle (u32_E * data, u32 log_len, u32 log_stride, u32 deg, mont256::Params* param, u32 group_sz, u32 * roots, const u32 * inv_n, const u32 * zeta) {
        using barrier = cuda::barrier<cuda::thread_scope_block>;
        #pragma nv_diag_suppress static_var_with_dynamic_init
        __shared__  barrier bar;

        if (threadIdx.x == 0) {
            init(&bar, blockDim.x); // Initialize the barrier with expected arrival count
        }
        __syncthreads();

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

        u32 log_end_stride = (log_stride - deg + 1);
        u32 end_stride = 1 << log_end_stride; //stride of the last butterfly
        u32 end_pair_stride = 1 << (log_len - log_stride - 2 + deg); // the stride between the last pair of butterfly

        // each segment is independent
        // uint segment_stride = end_pair_stride << 1; // the distance between two segment
        u32 log_segment_num = (log_len - log_stride - 1 - deg); // log of # of blocks in a segment
        
        u32 segment_start = (index >> log_segment_num) << (log_segment_num + (deg << 1)); // segment_start = index / segment_num * segment_stride;

        u32 segment_id = index & ((1 << log_segment_num) - 1); // segment_id = index & (segment_num - 1);
        
        u32 subblock_sz = 1 << (deg - 1); // # of neighbouring butterfly in the last round
        u32 subblock_offset = (segment_id >> log_end_stride) << (deg + log_end_stride); // subblock_offset = (segment_id / (end_stride)) * (2 * subblock_sz * end_stride);
        u32 subblock_id = segment_id & (end_stride - 1);

        data += ((u64)(segment_start + subblock_offset + subblock_id)) * WORDS; // use u64 to avoid overflow

        const u32 io_id = lid & (io_group - 1);
        const u32 lid_start = lid - io_id;
        const u32 cur_io_group = io_group < lsize ? io_group : lsize;

        mont256::Element a, b, c, d;

        // Read data
        if (cur_io_group == io_group) {
            for (int i = lid_start; i != lid_start + io_group; i ++) {
                if (io_id < WORDS) {
                    u32 group_offset = (i >> (deg - 1)) << (log_len - log_stride - 1);
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_offset + (group_id << (log_end_stride));

                    thread_data[i - lid_start] = data[(gpos) * WORDS + io_id];
                }
            }

            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);
            a = mont256::Element::load(thread_data);
            __syncwarp();

            for (int i = lid_start; i != lid_start + io_group; i ++) {
                if (io_id < WORDS) {
                    u32 group_offset = (i >> (deg - 1)) << (log_len - log_stride - 1);
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_offset + (group_id << (log_end_stride));

                    thread_data[i - lid_start] = data[(gpos+ (end_stride << (deg - 1))) * WORDS + io_id];
                }
            }

            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);
            b = mont256::Element::load(thread_data);
            __syncwarp();

            for (int i = lid_start; i != lid_start + io_group; i ++) {
                if (io_id < WORDS) {
                    u32 group_offset = (i >> (deg - 1)) << (log_len - log_stride - 1);
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_offset + (group_id << (log_end_stride));

                    thread_data[i - lid_start] = data[(gpos+ end_pair_stride) * WORDS + io_id];
                }
            }

            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);
            c = mont256::Element::load(thread_data);
            __syncwarp();

            for (int i = lid_start; i != lid_start + io_group; i ++) {
                if (io_id < WORDS) {
                    u32 group_offset = (i >> (deg - 1)) << (log_len - log_stride - 1);
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_offset + (group_id << (log_end_stride));

                    thread_data[i - lid_start] = data[(gpos + (end_stride << (deg - 1)) + end_pair_stride) * WORDS + io_id];
                }
            }

            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);
            d = mont256::Element::load(thread_data);
            __syncwarp();
        } else {
            u32 group_offset = (lid >> (deg - 1)) << (log_len - log_stride - 1);
            u32 group_id = lid & (subblock_sz - 1);
            u64 gpos = group_offset + (group_id << (log_end_stride));

            a = mont256::Element::load(data + (gpos) * WORDS);
            b = mont256::Element::load(data + (gpos + (end_stride << (deg - 1))) * WORDS);
            c = mont256::Element::load(data + (gpos + end_pair_stride) * WORDS);
            d = mont256::Element::load(data + (gpos + (end_stride << (deg - 1)) + end_pair_stride) * WORDS);
        }
        
        auto env = mont256::Env(*param);

        barrier::arrival_token token = bar.arrive(); /* this thread arrives. Arrival does not block a thread */

        const u32 pqshift = log_len - 1 - log_stride;

        for (u32 i = 0; i < deg; i++) {
            if (i != 0) {
                u32 lanemask = 1 << (deg - i - 1);
                mont256::Element tmp, tmp1;
                tmp = ((lid / lanemask) & 1) ? a : b;
                tmp1 = ((lid / lanemask) & 1) ? c : d;
                tmp.n.c0 = __shfl_xor_sync(0xffffffff, tmp.n.c0, lanemask);
                tmp.n.c1 = __shfl_xor_sync(0xffffffff, tmp.n.c1, lanemask);
                tmp.n.c2 = __shfl_xor_sync(0xffffffff, tmp.n.c2, lanemask);
                tmp.n.c3 = __shfl_xor_sync(0xffffffff, tmp.n.c3, lanemask);
                tmp.n.c4 = __shfl_xor_sync(0xffffffff, tmp.n.c4, lanemask);
                tmp.n.c5 = __shfl_xor_sync(0xffffffff, tmp.n.c5, lanemask);
                tmp.n.c6 = __shfl_xor_sync(0xffffffff, tmp.n.c6, lanemask);
                tmp.n.c7 = __shfl_xor_sync(0xffffffff, tmp.n.c7, lanemask);

                tmp1.n.c0 = __shfl_xor_sync(0xffffffff, tmp1.n.c0, lanemask);
                tmp1.n.c1 = __shfl_xor_sync(0xffffffff, tmp1.n.c1, lanemask);
                tmp1.n.c2 = __shfl_xor_sync(0xffffffff, tmp1.n.c2, lanemask);
                tmp1.n.c3 = __shfl_xor_sync(0xffffffff, tmp1.n.c3, lanemask);
                tmp1.n.c4 = __shfl_xor_sync(0xffffffff, tmp1.n.c4, lanemask);
                tmp1.n.c5 = __shfl_xor_sync(0xffffffff, tmp1.n.c5, lanemask);
                tmp1.n.c6 = __shfl_xor_sync(0xffffffff, tmp1.n.c6, lanemask);
                tmp1.n.c7 = __shfl_xor_sync(0xffffffff, tmp1.n.c7, lanemask);

                if ((lid / lanemask) & 1) a = tmp, c = tmp1;
                else b = tmp, d = tmp1;
            }

            auto tmp1 = a;
            auto tmp2 = c;

            a = env.add(a, b);
            c = env.add(c, d);
            b = env.sub(tmp1, b);
            d = env.sub(tmp2, d);

            u32 bit = subblock_sz >> i;
            u64 di = (lid & (bit - 1)) * end_stride + subblock_id;

            if (di != 0) {
                auto w = mont256::Element::load(roots + (di << i << pqshift) * WORDS);
                b = env.mul(b, w);
                d = env.mul(d, w);
            }
        }

        if (inverse) {
            auto inv = mont256::Element::load(inv_n);
            a = env.mul(a, inv);
            b = env.mul(b, inv);
            c = env.mul(c, inv);
            d = env.mul(d, inv);
            if (process) {
                u32 p;
                u32 second_half_l, gap;
                u32 lid_l;
                u32 group_offset, group_id;
                u64 gpos;
                mont256::Element z;
                u32 id;

                p = __brev((lid << 1)) >> (32 - (deg << 1));
                second_half_l = (p >= lsize * 2);

                lid_l = (p - second_half_l * lsize * 2);
                gap = lid_l & 1;
                lid_l = lid_l >> 1;
                

                group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
                group_id = lid_l & (subblock_sz - 1);
                gpos = group_offset + (group_id << (log_end_stride + 1));
                
                id = ((segment_start + subblock_offset + subblock_id) + gpos + second_half_l * end_pair_stride + gap * end_stride) % 3;
                if (id != 0) {
                    z = mont256::Element::load(zeta + WORDS * (id - 1));
                    a = env.mul(a, z);
                }

                p = __brev((lid << 1) + 1) >> (32 - (deg << 1));
                second_half_l = (p >= lsize * 2);

                lid_l = (p - second_half_l * lsize * 2);
                gap = lid_l & 1;
                lid_l = lid_l >> 1;
                

                group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
                group_id = lid_l & (subblock_sz - 1);
                gpos = group_offset + (group_id << (log_end_stride + 1));
                
                id = ((segment_start + subblock_offset + subblock_id) + gpos + second_half_l * end_pair_stride + gap * end_stride) % 3;
                if (id != 0) {
                    z = mont256::Element::load(zeta + WORDS * (id - 1));
                    b = env.mul(b, z);
                }

                p = __brev((lid << 1) + lsize * 2) >> (32 - (deg << 1));
                second_half_l = (p >= lsize * 2);

                lid_l = (p - second_half_l * lsize * 2);
                gap = lid_l & 1;
                lid_l = lid_l >> 1;
                

                group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
                group_id = lid_l & (subblock_sz - 1);
                gpos = group_offset + (group_id << (log_end_stride + 1));
                
                id = ((segment_start + subblock_offset + subblock_id) + gpos + second_half_l * end_pair_stride + gap * end_stride) % 3;
                if (id != 0) {
                    z = mont256::Element::load(zeta + WORDS * (id - 1));
                    c = env.mul(c, z);
                }

                p = __brev((lid << 1) + lsize * 2 + 1) >> (32 - (deg << 1));
                second_half_l = (p >= lsize * 2);

                lid_l = (p - second_half_l * lsize * 2);
                gap = lid_l & 1;
                lid_l = lid_l >> 1;
                

                group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
                group_id = lid_l & (subblock_sz - 1);
                gpos = group_offset + (group_id << (log_end_stride + 1));
                
                id = ((segment_start + subblock_offset + subblock_id) + gpos + second_half_l * end_pair_stride + gap * end_stride) % 3;
                if (id != 0) {
                    z = mont256::Element::load(zeta + WORDS * (id - 1));
                    d = env.mul(d, z);
                }
            }
        }
        
        bar.wait(std::move(token)); /* wait for all threads participating in the barrier to complete bar.arrive()*/

        // Write back
        if (cur_io_group == io_group) {
            a.store(thread_data);
            WarpExchangeT(temp_storage[warp_id]).BlockedToStriped(thread_data, thread_data);
            __syncwarp();

            for (int ti = lid_start; ti != lid_start + io_group; ti ++) {
                u32 p;
                u32 second_half_l, gap;
                u32 lid_l;
                u32 group_offset, group_id;
                u64 gpos;

                p = __brev((ti << 1)) >> (32 - (deg << 1));
                second_half_l = (p >= lsize * 2);

                lid_l = (p - second_half_l * lsize * 2);
                gap = lid_l & 1;
                lid_l = lid_l >> 1;
                

                group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
                group_id = lid_l & (subblock_sz - 1);
                gpos = group_offset + (group_id << (log_end_stride + 1));
                
                if (io_id < WORDS) {
                    data[(gpos + second_half_l * end_pair_stride + gap * end_stride) * WORDS + io_id] = thread_data[ti - lid_start];
                }
            }

            b.store(thread_data);
            WarpExchangeT(temp_storage[warp_id]).BlockedToStriped(thread_data, thread_data);
            __syncwarp();

            for (int ti = lid_start; ti != lid_start + io_group; ti ++) {
                u32 p;
                u32 second_half_l, gap;
                u32 lid_l;
                u32 group_offset, group_id;
                u64 gpos;

                p = __brev((ti << 1) + 1) >> (32 - (deg << 1));
                second_half_l = (p >= lsize * 2);

                lid_l = (p - second_half_l * lsize * 2);
                gap = lid_l & 1;
                lid_l = lid_l >> 1;
                

                group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
                group_id = lid_l & (subblock_sz - 1);
                gpos = group_offset + (group_id << (log_end_stride + 1));
                
                if (io_id < WORDS) {
                    data[(gpos + second_half_l * end_pair_stride + gap * end_stride) * WORDS + io_id] = thread_data[ti - lid_start];
                }
            }

            c.store(thread_data);
            WarpExchangeT(temp_storage[warp_id]).BlockedToStriped(thread_data, thread_data);
            __syncwarp();

            for (int ti = lid_start; ti != lid_start + io_group; ti ++) {
                u32 p;
                u32 second_half_l, gap;
                u32 lid_l;
                u32 group_offset, group_id;
                u64 gpos;

                p = __brev((ti << 1) + lsize * 2) >> (32 - (deg << 1));
                second_half_l = (p >= lsize * 2);

                lid_l = (p - second_half_l * lsize * 2);
                gap = lid_l & 1;
                lid_l = lid_l >> 1;
                

                group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
                group_id = lid_l & (subblock_sz - 1);
                gpos = group_offset + (group_id << (log_end_stride + 1));
                
                if (io_id < WORDS) {
                    data[(gpos + second_half_l * end_pair_stride + gap * end_stride) * WORDS + io_id] = thread_data[ti - lid_start];
                }
            }

            d.store(thread_data);
            WarpExchangeT(temp_storage[warp_id]).BlockedToStriped(thread_data, thread_data);

            for (int ti = lid_start; ti != lid_start + io_group; ti ++) {
                u32 p;
                u32 second_half_l, gap;
                u32 lid_l;
                u32 group_offset, group_id;
                u64 gpos;

                p = __brev((ti << 1) + 1 + lsize * 2) >> (32 - (deg << 1));
                second_half_l = (p >= lsize * 2);

                lid_l = (p - second_half_l * lsize * 2);
                gap = lid_l & 1;
                lid_l = lid_l >> 1;
                

                group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
                group_id = lid_l & (subblock_sz - 1);
                gpos = group_offset + (group_id << (log_end_stride + 1));
                
                if (io_id < WORDS) {
                    data[(gpos + second_half_l * end_pair_stride + gap * end_stride) * WORDS + io_id] = thread_data[ti - lid_start];
                }
            }

        } else {
            
            u32 p;
            u32 second_half_l, gap;
            u32 lid_l;
            u32 group_offset, group_id;
            u64 gpos;

            p = __brev((lid << 1)) >> (32 - (deg << 1));
            second_half_l = (p >= lsize * 2);

            lid_l = (p - second_half_l * lsize * 2);
            gap = lid_l & 1;
            lid_l = lid_l >> 1;
            

            group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
            group_id = lid_l & (subblock_sz - 1);
            gpos = group_offset + (group_id << (log_end_stride + 1));
            
            a.store(data + (gpos + second_half_l * end_pair_stride + gap * end_stride) * WORDS);

            p = __brev((lid << 1) + 1) >> (32 - (deg << 1));
            second_half_l = (p >= lsize * 2);

            lid_l = (p - second_half_l * lsize * 2);
            gap = lid_l & 1;
            lid_l = lid_l >> 1;
            

            group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
            group_id = lid_l & (subblock_sz - 1);
            gpos = group_offset + (group_id << (log_end_stride + 1));
            
            b.store(data + (gpos + second_half_l * end_pair_stride + gap * end_stride) * WORDS);

            p = __brev((lid << 1) + lsize * 2) >> (32 - (deg << 1));
            second_half_l = (p >= lsize * 2);

            lid_l = (p - second_half_l * lsize * 2);
            gap = lid_l & 1;
            lid_l = lid_l >> 1;
            

            group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
            group_id = lid_l & (subblock_sz - 1);
            gpos = group_offset + (group_id << (log_end_stride + 1));
            
            c.store(data + (gpos + second_half_l * end_pair_stride + gap * end_stride) * WORDS);

            p = __brev((lid << 1) + lsize * 2 + 1) >> (32 - (deg << 1));
            second_half_l = (p >= lsize * 2);

            lid_l = (p - second_half_l * lsize * 2);
            gap = lid_l & 1;
            lid_l = lid_l >> 1;
            

            group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
            group_id = lid_l & (subblock_sz - 1);
            gpos = group_offset + (group_id << (log_end_stride + 1));
            
            d.store(data + (gpos + second_half_l * end_pair_stride + gap * end_stride) * WORDS);
        }
    }

    template <u32 WORDS, u32 io_group, bool process>
    __global__ void SSIP_NTT_stage1_warp_no_twiddle (u32_E * x, u32 log_len, u32 log_stride, u32 deg, mont256::Params* param, u32 group_sz, u32 * roots, const u32 * zeta, u32 start_len) {
        extern __shared__ u32_E s[];

        const u32 lid = threadIdx.x & (group_sz - 1);
        const u32 lsize = group_sz;
        const u32 group_id = threadIdx.x / group_sz;
        const u32 group_num = blockDim.x / group_sz;
        const u32 index = blockIdx.x * group_num + group_id;

        auto u = s + group_id * ((1 << deg) + 1) * WORDS;

        const u32 lgp = log_stride - deg + 1;
        const u32 end_stride = 1 << lgp; //stride of the last butterfly

        // each segment is independent
        const u32 segment_start = (index >> lgp) << (lgp + deg);
        const u32 segment_id = index & (end_stride - 1);
        
        const u32 subblock_sz = 1 << (deg - 1); // # of neighbouring butterfly in the last round

        x += ((u64)(segment_start + segment_id)) * WORDS; // use u64 to avoid overflow

        const u32 io_id = lid & (io_group - 1);
        const u32 lid_start = lid - io_id;
        const u32 shared_read_stride = (lsize << 1) + 1;
        const u32 cur_io_group = io_group < lsize ? io_group : lsize;
        const u32 io_per_thread = io_group / cur_io_group;

        const u32 io_group_id = lid / cur_io_group;
        int io_st, io_ed, io_stride;

        if ((io_group_id / 2) & 1) {
            io_st = lid_start;
            io_ed = lid_start + cur_io_group;
            io_stride = 1;
        } else {
            io_st = lid_start + cur_io_group - 1;
            io_ed = ((int)lid_start) - 1;
            io_stride = -1;
        }
        // Read data
        for (int i = io_st; i != io_ed; i += io_stride) {
            for (u32 j = 0; j < io_per_thread; j++) {
                u32 io = io_id + j * cur_io_group;
                if (io < WORDS) {
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_id << (lgp + 1);
                    if (!process) {
                        u[(i << 1) + io * shared_read_stride] = x[gpos * WORDS + io];
                        u[(i << 1) + 1 + io * shared_read_stride] = x[(gpos + end_stride) * WORDS + io];
                    } else {
                        u[(i << 1) + io * shared_read_stride] = gpos >= start_len ? 0 : x[gpos * WORDS + io];
                        u[(i << 1) + 1 + io * shared_read_stride] = gpos + end_stride >= start_len ? 0 : x[(gpos + end_stride) * WORDS + io];
                    }
                }
            }
        }

        auto env = mont256::Env(*param);

        __syncthreads();

        const u32 pqshift = log_len - 1 - log_stride;

        for(u32 rnd = 0; rnd < deg; rnd += 6) {
            u32 sub_deg = min(6, deg - rnd);
            u32 warp_sz = 1 << (sub_deg - 1);
            u32 warp_id = lid / warp_sz;
            
            u32 lgp = deg - rnd - sub_deg;
            u32 end_stride_warp = 1 << lgp;

            u32 segment_start_warp = (warp_id >> lgp) << (lgp + sub_deg);
            u32 segment_id_warp = warp_id & (end_stride_warp - 1);
            
            u32 laneid = lid & (warp_sz - 1);

            u32 bit = subblock_sz >> rnd;
            u32 i0 = segment_start_warp + segment_id_warp + laneid * end_stride_warp;
            u32 i1 = i0 + bit;

            auto a = mont256::Element::load(u + i0, shared_read_stride);
            auto b = mont256::Element::load(u + i1, shared_read_stride);

            if (process) if(rnd == 0) {
                auto ida = ((segment_start + segment_id) + i0 * end_stride);
                auto idb = (ida + (1 << (log_len - 1)));
                if (ida % 3 != 0 && ida < start_len) {
                    auto z = mont256::Element::load(zeta + (ida % 3 - 1) * WORDS);
                    a = env.mul(a, z);
                }
                if (idb % 3 != 0 && idb < start_len) {
                    auto z = mont256::Element::load(zeta + (idb % 3 - 1) * WORDS);
                    b = env.mul(b, z);
                }
            }

            for (u32 i = 0; i < sub_deg; i++) {
                if (i != 0) {
                    u32 lanemask = 1 << (sub_deg - i - 1);
                    mont256::Element tmp;
                    tmp = ((lid / lanemask) & 1) ? a : b;
                    tmp.n.c0 = __shfl_xor_sync(0xffffffff, tmp.n.c0, lanemask);
                    tmp.n.c1 = __shfl_xor_sync(0xffffffff, tmp.n.c1, lanemask);
                    tmp.n.c2 = __shfl_xor_sync(0xffffffff, tmp.n.c2, lanemask);
                    tmp.n.c3 = __shfl_xor_sync(0xffffffff, tmp.n.c3, lanemask);
                    tmp.n.c4 = __shfl_xor_sync(0xffffffff, tmp.n.c4, lanemask);
                    tmp.n.c5 = __shfl_xor_sync(0xffffffff, tmp.n.c5, lanemask);
                    tmp.n.c6 = __shfl_xor_sync(0xffffffff, tmp.n.c6, lanemask);
                    tmp.n.c7 = __shfl_xor_sync(0xffffffff, tmp.n.c7, lanemask);
                    if ((lid / lanemask) & 1) a = tmp;
                    else b = tmp;
                }

                auto tmp = a;
                a = env.add(a, b);
                b = env.sub(tmp, b);
                u32 bit = (1 << sub_deg) >> (i + 1);
                u64 di = ((lid & (bit - 1)) * end_stride_warp + segment_id_warp) * end_stride + segment_id;

                if (di != 0) {
                    auto w = mont256::Element::load(roots + (di << (rnd + i) << pqshift) * WORDS);
                    b = env.mul(b, w);
                }
            }            

            i0 = segment_start_warp + segment_id_warp + laneid * 2 * end_stride_warp;
            i1 = i0 + end_stride_warp;
            a.store(u + i0, shared_read_stride);
            b.store(u + i1, shared_read_stride);

            __syncthreads();
        }

        // Write back
        for (int i = io_st; i != io_ed; i += io_stride) {
            for (u32 j = 0; j < io_per_thread; j++) {
                u32 io = io_id + j * cur_io_group;
                if (io < WORDS) {
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_id << (lgp + 1);
                    x[gpos * WORDS + io] = u[(i << 1) + io * shared_read_stride];
                    x[(gpos + end_stride) * WORDS + io] = u[(i << 1) + 1 + io * shared_read_stride];
                }
            }
        }
    }

    template <u32 WORDS, u32 io_group>
    __global__ void SSIP_NTT_stage1_warp_no_smem_no_twiddle (u32_E * data, u32 log_len, u32 log_stride, u32 deg, mont256::Params* param, u32 group_sz, u32 * roots) {
        using barrier = cuda::barrier<cuda::thread_scope_block>;
        #pragma nv_diag_suppress static_var_with_dynamic_init
        __shared__  barrier bar;

        if (threadIdx.x == 0) {
            init(&bar, blockDim.x); // Initialize the barrier with expected arrival count
        }
        __syncthreads();

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

        mont256::Element a, b;

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
            a = mont256::Element::load(thread_data);
            __syncwarp();

            for (int i = lid_start; i != lid_start + io_group; i ++) {
                if (io_id < WORDS) {
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_id << (lgp);

                    thread_data[i - lid_start] = data[(gpos+ (end_stride << (deg - 1))) * WORDS + io_id];
                }
            }

            WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);
            b = mont256::Element::load(thread_data);
            __syncwarp();

        } else {
            u32 group_id = lid & (subblock_sz - 1);
            u64 gpos = group_id << (lgp);

            a = mont256::Element::load(data + (gpos) * WORDS);
            b = mont256::Element::load(data + (gpos + (end_stride << (deg - 1))) * WORDS);
        }

        // printf("threadid: %d, a: %d, b: %d\n", threadIdx.x, a.n.c0, b.n.c0);
        
        auto env = mont256::Env(*param);

        barrier::arrival_token token = bar.arrive(); /* this thread arrives. Arrival does not block a thread */

        const u32 pqshift = log_len - 1 - log_stride;

        for (u32 i = 0; i < deg; i++) {
            if (i != 0) {
                u32 lanemask = 1 << (deg - i - 1);
                mont256::Element tmp;
                tmp = ((lid / lanemask) & 1) ? a : b;

                tmp.n.c0 = __shfl_xor_sync(0xffffffff, tmp.n.c0, lanemask);
                tmp.n.c1 = __shfl_xor_sync(0xffffffff, tmp.n.c1, lanemask);
                tmp.n.c2 = __shfl_xor_sync(0xffffffff, tmp.n.c2, lanemask);
                tmp.n.c3 = __shfl_xor_sync(0xffffffff, tmp.n.c3, lanemask);
                tmp.n.c4 = __shfl_xor_sync(0xffffffff, tmp.n.c4, lanemask);
                tmp.n.c5 = __shfl_xor_sync(0xffffffff, tmp.n.c5, lanemask);
                tmp.n.c6 = __shfl_xor_sync(0xffffffff, tmp.n.c6, lanemask);
                tmp.n.c7 = __shfl_xor_sync(0xffffffff, tmp.n.c7, lanemask);

                if ((lid / lanemask) & 1) a = tmp;
                else b = tmp;
            }

            auto tmp = a;

            a = env.add(a, b);
            b = env.sub(tmp, b);

            u32 bit = subblock_sz >> i;
            u64 di = (lid & (bit - 1)) * end_stride + segment_id;

            if (di != 0) {
                auto w = mont256::Element::load(roots + (di << i << pqshift) * WORDS);
                b = env.mul(b, w);
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

        template <u32 WORDS, u32 io_group>
    __global__ void SSIP_NTT_stage1_warp_no_twiddle_coalesced_roots (u32_E * x, u32 log_len, u32 log_stride, u32 deg, mont256::Params* param, u32 group_sz, u32 * roots) {
        extern __shared__ u32_E s[];

        constexpr int warp_threads = io_group;
        constexpr int items_per_thread = io_group;
        const int warp_id_exchange = static_cast<int>(threadIdx.x) / warp_threads;
        u32 thread_data[io_group];

        // Specialize WarpExchange for a virtual warp of a threads owning b integer items each
        using WarpExchangeT = cub::WarpExchange<u32, items_per_thread, warp_threads>;

        // Allocate shared memory for WarpExchange
        typename WarpExchangeT::TempStorage *temp_storage = (typename WarpExchangeT::TempStorage*) s;

        const u32 lid = threadIdx.x & (group_sz - 1);
        const u32 lsize = group_sz;
        const u32 group_id = threadIdx.x / group_sz;
        const u32 group_num = blockDim.x / group_sz;
        const u32 index = blockIdx.x * group_num + group_id;

        auto u = s + (sizeof(WarpExchangeT::TempStorage) / sizeof(u32) * (blockDim.x / warp_threads)) + group_id * ((1 << deg) + 1) * WORDS;

        const u32 lgp = log_stride - deg + 1;
        const u32 end_stride = 1 << lgp; //stride of the last butterfly

        // each segment is independent
        const u32 segment_start = (index >> lgp) << (lgp + deg);
        const u32 segment_id = index & (end_stride - 1);
        
        const u32 subblock_sz = 1 << (deg - 1); // # of neighbouring butterfly in the last round

        x += ((u64)(segment_start + segment_id)) * WORDS; // use u64 to avoid overflow

        const u32 io_id = lid & (io_group - 1);
        const u32 lid_start = lid - io_id;
        const u32 shared_read_stride = (lsize << 1) + 1;
        const u32 cur_io_group = io_group < lsize ? io_group : lsize;
        const u32 io_per_thread = io_group / cur_io_group;

        const u32 io_group_id = lid / cur_io_group;
        int io_st, io_ed, io_stride;

        if ((io_group_id / 2) & 1) {
            io_st = lid_start;
            io_ed = lid_start + cur_io_group;
            io_stride = 1;
        } else {
            io_st = lid_start + cur_io_group - 1;
            io_ed = ((int)lid_start) - 1;
            io_stride = -1;
        }
        // Read data
        for (int i = io_st; i != io_ed; i += io_stride) {
            for (u32 j = 0; j < io_per_thread; j++) {
                u32 io = io_id + j * cur_io_group;
                if (io < WORDS) {
                    u32 group_id = i & (subblock_sz - 1);
                    u32 gpos = group_id << (lgp + 1);
                    u[(i << 1) + io * shared_read_stride] = x[gpos * WORDS + io];
                    u[(i << 1) + 1 + io * shared_read_stride] = x[(gpos + end_stride) * WORDS + io];
                }
            }
        }

        auto env = mont256::Env(*param);

        __syncthreads();

        const u32 pqshift = log_len - 1 - log_stride;
        
        for(u32 rnd = 0; rnd < deg; rnd += 6) {
            u32 sub_deg = min(6, deg - rnd);
            u32 warp_sz = 1 << (sub_deg - 1);
            u32 warp_id = lid / warp_sz;
            
            u32 lgp = deg - rnd - sub_deg;
            u32 end_stride_warp = 1 << lgp;

            u32 segment_start = (warp_id >> lgp) << (lgp + sub_deg);
            u32 segment_id_warp = warp_id & (end_stride_warp - 1);
            
            u32 laneid = lid & (warp_sz - 1);

            u32 gap = subblock_sz >> rnd;
            u32 i0 = segment_start + segment_id_warp + laneid * end_stride_warp;
            u32 i1 = i0 + gap;

            auto a = mont256::Element::load(u + i0, shared_read_stride);
            auto b = mont256::Element::load(u + i1, shared_read_stride);

            mont256::Element w;

            auto tmp = a;
            a = env.add(a, b);
            b = env.sub(tmp, b);
            u32 bit = (1 << sub_deg) >> 1;
            u64 di = ((lid & (bit - 1)) * end_stride_warp + segment_id_warp) * end_stride + segment_id;

            if (io_group == cur_io_group) {
                for (int i = lid_start; i < lid_start + io_group; i++) {
                    if (io_id < WORDS) {
                        u32 warp_id = i / warp_sz;
                        u32 segment_id_warp = warp_id & (end_stride_warp - 1);
                        u64 pos = ((i & (bit - 1)) * end_stride_warp + segment_id_warp) * end_stride + segment_id;
                        pos = pos << rnd << pqshift;
                        thread_data[i - lid_start] = roots[pos * WORDS + io_id];
                    }
                }

                WarpExchangeT(temp_storage[warp_id_exchange]).StripedToBlocked(thread_data, thread_data);
                w = mont256::Element::load(thread_data);
                __syncwarp();
            } else {
                w = mont256::Element::load(roots + (di << (rnd) << pqshift) * WORDS);
            }

            if (di != 0) {
                b = env.mul(b, w);
            }

            for (u32 i = 1; i < sub_deg; i++) {
                u32 lanemask = 1 << (sub_deg - i - 1);
                mont256::Element tmp;
                tmp = ((lid / lanemask) & 1) ? a : b;
                tmp.n.c0 = __shfl_xor_sync(0xffffffff, tmp.n.c0, lanemask);
                tmp.n.c1 = __shfl_xor_sync(0xffffffff, tmp.n.c1, lanemask);
                tmp.n.c2 = __shfl_xor_sync(0xffffffff, tmp.n.c2, lanemask);
                tmp.n.c3 = __shfl_xor_sync(0xffffffff, tmp.n.c3, lanemask);
                tmp.n.c4 = __shfl_xor_sync(0xffffffff, tmp.n.c4, lanemask);
                tmp.n.c5 = __shfl_xor_sync(0xffffffff, tmp.n.c5, lanemask);
                tmp.n.c6 = __shfl_xor_sync(0xffffffff, tmp.n.c6, lanemask);
                tmp.n.c7 = __shfl_xor_sync(0xffffffff, tmp.n.c7, lanemask);
                if ((lid / lanemask) & 1) a = tmp;
                else b = tmp;

                tmp = a;
                a = env.add(a, b);
                b = env.sub(tmp, b);
                u32 bit = (1 << sub_deg) >> (i + 1);
                u64 di = ((lid & (bit - 1)) * end_stride_warp + segment_id_warp) * end_stride + segment_id;

                if (io_group == cur_io_group) {
                    for (int ti = lid_start; ti < lid_start + io_group; ti++) {
                        if (io_id < WORDS) {
                            u32 warp_id = ti / warp_sz;
                            u32 segment_id_warp = warp_id & (end_stride_warp - 1);
                            u64 pos = ((ti & (bit - 1)) * end_stride_warp + segment_id_warp) * end_stride + segment_id;
                            pos = pos << (rnd + i) << pqshift;
                            thread_data[ti - lid_start] = roots[pos * WORDS + io_id];
                        }
                    }

                    WarpExchangeT(temp_storage[warp_id_exchange]).StripedToBlocked(thread_data, thread_data);
                    w = mont256::Element::load(thread_data);
                    __syncwarp();
                } else {
                    w = mont256::Element::load(roots + (di << (rnd + i) << pqshift) * WORDS);
                }
                if (di != 0) {
                    b = env.mul(b, w);
                }
            }            

            i0 = segment_start + segment_id_warp + laneid * 2 * end_stride_warp;
            i1 = i0 + end_stride_warp;
            a.store(u + i0, shared_read_stride);
            b.store(u + i1, shared_read_stride);

            __syncthreads();
        }

        // Write back
        for (int i = io_st; i != io_ed; i += io_stride) {
            for (u32 j = 0; j < io_per_thread; j++) {
                u32 io = io_id + j * cur_io_group;
                if (io < WORDS) {
                    u32 group_id = i & (subblock_sz - 1);
                    u32 gpos = group_id << (lgp + 1);
                    x[gpos * WORDS + io] = u[(i << 1) + io * shared_read_stride];
                    x[(gpos + end_stride) * WORDS + io] = u[(i << 1) + 1 + io * shared_read_stride];
                }
            }
        }
    }

    #define CONVERT_SADDR(x)  (((x) >> 5) * 33 + ((x) & 31))
    template <u32 WORDS, u32 io_group>
    __global__ void SSIP_NTT_stage1_warp_no_twiddle_opt_smem (u32_E * x, u32 log_len, u32 log_stride, u32 deg, mont256::Params* param, u32 group_sz, u32 * roots) {
        extern __shared__ u32_E s[];
        // column-major in shared memory
        // data patteren:
        // a0_word0 a1_word0 a2_word0 a3_word0 ... an_word0 [empty] a0_word1 a1_word1 a2_word1 a3_word1 ...
        // for coleasced read, we need to read a0_word0, a0_word1, a0_word2, ... a0_wordn in a single read
        // so thread [0,WORDS) will read a0_word0 a0_word1 a0_word2 ... 
        // then a1_word0 a1_word1 a1_word2 ... 
        // so we need the empty space is for padding to avoid bank conflict during read because n is likely to be 32k

        const u32 lid = threadIdx.x & (group_sz - 1);
        const u32 lsize = group_sz;
        const u32 group_id = threadIdx.x / group_sz;
        const u32 group_num = blockDim.x / group_sz;
        const u32 index = blockIdx.x * group_num + group_id;

        auto u = s + group_id * (CONVERT_SADDR(((1 << deg) + 1) * WORDS));

        const u32 lgp = log_stride - deg + 1;
        const u32 end_stride = 1 << lgp; //stride of the last butterfly

        // each segment is independent
        const u32 segment_start = (index >> lgp) << (lgp + deg);
        const u32 segment_id = index & (end_stride - 1);
        
        const u32 subblock_sz = 1 << (deg - 1); // # of neighbouring butterfly in the last round

        x += ((u64)(segment_start + segment_id)) * WORDS; // use u64 to avoid overflow

        const u32 io_id = lid & (io_group - 1);
        const u32 lid_start = lid - io_id;
        const u32 shared_read_stride = CONVERT_SADDR(lsize << 1) + 1;
        const u32 cur_io_group = io_group < lsize ? io_group : lsize;
        const u32 io_per_thread = io_group / cur_io_group;

        const u32 io_group_id = lid / cur_io_group;
        int io_st, io_ed, io_stride;

        if ((io_group_id / 2) & 1) {
            io_st = lid_start;
            io_ed = lid_start + cur_io_group;
            io_stride = 1;
        } else {
            io_st = lid_start + cur_io_group - 1;
            io_ed = ((int)lid_start) - 1;
            io_stride = -1;
        }
        // Read data
        for (int i = io_st; i != io_ed; i += io_stride) {
            for (u32 j = 0; j < io_per_thread; j++) {
                u32 io = io_id + j * cur_io_group;
                if (io < WORDS) {
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_id << (lgp + 1);

                    u[CONVERT_SADDR(i << 1) + io * shared_read_stride] = x[gpos * WORDS + io];
                    u[CONVERT_SADDR((i << 1) + 1) + io * shared_read_stride] = x[(gpos + end_stride) * WORDS + io];

                    // printf("%d %d %d %d\n",(i << 1) + io * shared_read_stride, CONVERT_SADDR(i << 1) + io * shared_read_stride, CONVERT_SADDR((i << 1) + 1) + io * shared_read_stride, threadIdx.x);
                }
            }
        }

        auto env = mont256::Env(*param);

        __syncthreads();

        const u32 pqshift = log_len - 1 - log_stride;

        for(u32 rnd = 0; rnd < deg; rnd += 6) {
            u32 sub_deg = min(6, deg - rnd);
            u32 warp_sz = 1 << (sub_deg - 1);
            u32 warp_id = lid / warp_sz;
            
            u32 lgp = deg - rnd - sub_deg;
            u32 end_stride_warp = 1 << lgp;

            u32 segment_start = (warp_id >> lgp) << (lgp + sub_deg);
            u32 segment_id_warp = warp_id & (end_stride_warp - 1);
            
            u32 laneid = lid & (warp_sz - 1);

            u32 bit = subblock_sz >> rnd;
            u32 i0 = segment_start + segment_id_warp + laneid * end_stride_warp;
            u32 i1 = i0 + bit;

            auto a = mont256::Element::load(u + CONVERT_SADDR(i0), shared_read_stride);
            auto b = mont256::Element::load(u + CONVERT_SADDR(i1), shared_read_stride);

            for (u32 i = 0; i < sub_deg; i++) {
                if (i != 0) {
                    u32 lanemask = 1 << (sub_deg - i - 1);
                    mont256::Element tmp;
                    tmp = ((lid / lanemask) & 1) ? a : b;
                    tmp.n.c0 = __shfl_xor_sync(0xffffffff, tmp.n.c0, lanemask);
                    tmp.n.c1 = __shfl_xor_sync(0xffffffff, tmp.n.c1, lanemask);
                    tmp.n.c2 = __shfl_xor_sync(0xffffffff, tmp.n.c2, lanemask);
                    tmp.n.c3 = __shfl_xor_sync(0xffffffff, tmp.n.c3, lanemask);
                    tmp.n.c4 = __shfl_xor_sync(0xffffffff, tmp.n.c4, lanemask);
                    tmp.n.c5 = __shfl_xor_sync(0xffffffff, tmp.n.c5, lanemask);
                    tmp.n.c6 = __shfl_xor_sync(0xffffffff, tmp.n.c6, lanemask);
                    tmp.n.c7 = __shfl_xor_sync(0xffffffff, tmp.n.c7, lanemask);
                    if ((lid / lanemask) & 1) a = tmp;
                    else b = tmp;
                }

                auto tmp = a;
                a = env.add(a, b);
                b = env.sub(tmp, b);
                u32 bit = (1 << sub_deg) >> (i + 1);
                u64 di = ((lid & (bit - 1)) * end_stride_warp + segment_id_warp) * end_stride + segment_id;

                if (di != 0) {
                    auto w = mont256::Element::load(roots + (di << (rnd + i) << pqshift) * WORDS);
                    b = env.mul(b, w);
                }
            }            

            i0 = segment_start + segment_id_warp + laneid * 2 * end_stride_warp;
            i1 = i0 + end_stride_warp;
            a.store(u + CONVERT_SADDR(i0), shared_read_stride);
            b.store(u + CONVERT_SADDR(i1), shared_read_stride);

            __syncthreads();
        }

        // Write back
        for (int i = io_st; i != io_ed; i += io_stride) {
            for (u32 j = 0; j < io_per_thread; j++) {
                u32 io = io_id + j * cur_io_group;
                if (io < WORDS) {
                    u32 group_id = i & (subblock_sz - 1);
                    u64 gpos = group_id << (lgp + 1);

                    x[gpos * WORDS + io] = u[CONVERT_SADDR(i << 1) + io * shared_read_stride];
                    x[(gpos + end_stride) * WORDS + io] = u[CONVERT_SADDR((i << 1) + 1) + io * shared_read_stride];
                }
            }
        }
    }

    template <u32 WORDS>
    class self_sort_in_place_ntt : public best_ntt {
        u32 max_deg_stage1;
        u32 max_deg_stage2;
        u32 max_deg;
        const int log_len;
        const u64 len;
        mont256::Params *param, *param_d;
        mont256::Element unit;
        bool debug;
        u32_E *pq = nullptr, *pq_d; // Precalculated values for radix degrees up to `max_deg`
        u32_E *roots, *roots_d;
        const bool inverse; 
        const bool process;
        u32 * inv_n, * inv_n_d;
        u32 * zeta, * zeta_d;

        u32 get_deg (u32 deg_stage, u32 max_deg_stage) {
            u32 deg_per_round;
            for (u32 rounds = 1; ; rounds++) {
                deg_per_round = rounds == 1 ? deg_stage : (deg_stage - 1) / rounds + 1;
                if (deg_per_round <= max_deg_stage) break;
            }
            return deg_per_round;
        }

        static constexpr u32 log2_int(u32 x) {
            return 31 - std::countl_zero(x);
        }

        public:
        struct SSIP_config{
            enum stage1_config {
                stage1_naive,
                stage1_warp,
                stage1_warp_no_twiddle,
                stage1_warp_no_twiddle_no_smem,
                stage1_warp_no_twiddle_opt_smem,
            };
            stage1_config stage1_mode = stage1_warp_no_twiddle;
            bool stage1_coalesced_roots = false, stage2_coalesced_roots = false;
            enum stage2_config {
                stage2_naive,
                stage2_naive_2_per_thread,
                stage2_warp,
                stage2_warp_2_per_thread,
                stage2_warp_no_share,
                stage2_warp_2_per_thread_no_share,
                stage2_warp_no_twiddle_no_share,
            };
            stage2_config stage2_mode = stage2_warp_no_twiddle_no_share;

            u32 max_threads_stage1_log = 8;
            u32 max_threads_stage2_log = 8;
        };

        const SSIP_config config;
        float milliseconds = 0;

        self_sort_in_place_ntt(
            const mont256::Params param, 
            const u32* omega, u32 log_len, 
            bool debug, 
            u32 max_instance = 1,
            bool inverse = false, 
            bool process = false, 
            const u32 * inv_n = nullptr, 
            const u32 * zeta = nullptr, 
            SSIP_config config = SSIP_config()) 
        : best_ntt(max_instance), log_len(log_len), len(1ll << log_len), debug(debug)
        , config(config), inverse(inverse), process(process){
            bool success = true;
            cudaError_t first_err = cudaSuccess;

            u32 deg_stage1 = (log_len + 1) / 2;
            u32 deg_stage2 = log_len / 2;

            if (config.stage1_mode == SSIP_config::stage1_warp_no_twiddle_no_smem) {
                max_deg_stage1 = get_deg(deg_stage1, std::min(config.max_threads_stage1_log + 1, 6u)); // pure warp shuffle implementation
            } else {
                max_deg_stage1 = get_deg(deg_stage1, config.max_threads_stage1_log + 1);
            }

            if (config.stage2_mode == SSIP_config::stage2_naive_2_per_thread ||
            config.stage2_mode == SSIP_config::stage2_warp_2_per_thread ||
            config.stage2_mode == SSIP_config::stage2_warp_2_per_thread_no_share) {
                max_deg_stage2 = get_deg(deg_stage2, (config.max_threads_stage2_log + 1) / 2); // 2 elements per thread
            } else {            
                max_deg_stage2 = get_deg(deg_stage2, (config.max_threads_stage2_log + 2) / 2); // 4 elements per thread
            }

            max_deg = std::max(max_deg_stage1, max_deg_stage2);

            // Precalculate:
            auto env = mont256::Env::host_new(param);

            if (debug) {
                // unit = qpow(omega, (P - 1ll) / len)
                unit = env.host_from_number(mont256::Number::load(omega));
                auto exponent = mont256::Number::load(param.m);
                auto one = mont256::Number::zero();
                one.c0 = 1;
                exponent = exponent.host_sub(one);
                exponent = exponent.slr(log_len);
                unit = env.host_pow(unit, exponent);
            } else {
                unit = mont256::Element::load(omega);
            }

            if (config.stage1_mode == SSIP_config::stage1_naive ||
            config.stage1_mode == SSIP_config::stage1_warp ||
            (config.stage2_mode != SSIP_config::stage2_warp_no_twiddle_no_share)) {

                // pq: [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]

                CUDA_CHECK(cudaHostAlloc(&pq, (1 << max_deg >> 1) * sizeof(u32) * WORDS, cudaHostAllocDefault));
                memset(pq, 0, (1 << max_deg >> 1) * sizeof(u32) * WORDS);
                env.one().store(pq);
                auto twiddle = env.host_pow(unit, len >> max_deg);
                if (max_deg > 1) {
                    twiddle.store(pq + WORDS);
                    auto last = twiddle;
                    for (u32 i = 2; i < (1 << max_deg >> 1); i++) {
                        last = env.host_mul(last, twiddle);
                        last.store(pq + i * WORDS);
                    }
                }
            }

            if ((config.stage1_mode == SSIP_config::stage1_warp_no_twiddle_no_smem ||
            config.stage1_mode == SSIP_config::stage1_warp_no_twiddle_opt_smem ||
            config.stage1_mode == SSIP_config::stage1_warp_no_twiddle) &&
            (config.stage2_mode == SSIP_config::stage2_warp_no_twiddle_no_share)) {
                CUDA_CHECK(cudaHostAlloc(&roots, (u64)len / 2 * WORDS * sizeof(u32_E), cudaHostAllocDefault));
                gen_roots_cub<WORDS> gen;
                CUDA_CHECK(gen(roots, len / 2, unit, param));
            } else {
                CUDA_CHECK(cudaHostAlloc(&roots, (u64)len * WORDS * sizeof(u32_E), cudaHostAllocDefault));
                gen_roots_cub<WORDS> gen;
                CUDA_CHECK(gen(roots, len, unit, param));
            }

            CUDA_CHECK(cudaHostAlloc(&this->param, sizeof(mont256::Params), cudaHostAllocDefault));
            CUDA_CHECK(cudaMemcpy(this->param, &param, sizeof(mont256::Params), cudaMemcpyHostToHost));

            if (inverse) {
                assert(inv_n != nullptr);
                assert(config.stage2_mode == SSIP_config::stage2_config::stage2_warp_no_twiddle_no_share);
                CUDA_CHECK(cudaHostAlloc(&this->inv_n, sizeof(u32) * WORDS, cudaHostAllocDefault));
                CUDA_CHECK(cudaMemcpy(this->inv_n, inv_n, sizeof(u32) * WORDS, cudaMemcpyHostToHost));
            }

            if (process) {
                assert(zeta != nullptr);
                if (!inverse) {
                    assert(config.stage1_mode == SSIP_config::stage1_warp_no_twiddle);
                }
                CUDA_CHECK(cudaHostAlloc(&this->zeta, 2 * WORDS * sizeof(u32),cudaHostAllocDefault));
                CUDA_CHECK(cudaMemcpy(this->zeta, zeta, 2 * WORDS * sizeof(u32), cudaMemcpyHostToDevice));
            }

            if (!success) {
                std::cerr << "error occurred during gen_roots" << std::endl;
                throw cudaGetErrorString(first_err);
            }
        }

        ~self_sort_in_place_ntt() override {
            if (pq != nullptr) cudaFreeHost(pq);
            cudaFreeHost(roots);
            cudaFreeHost(param);
            if (inverse) cudaFreeHost(inv_n);
            if (process) cudaFreeHost(zeta);
            if (on_gpu) clean_gpu();
        }

        cudaError_t to_gpu(cudaStream_t stream = 0) override {
            std::unique_lock<std::shared_mutex> wlock(this->mtx);
            bool success = true;
            cudaError_t first_err = cudaSuccess;

            if (pq != nullptr) {
                CUDA_CHECK(cudaMallocAsync(&pq_d, (1 << max_deg >> 1) * sizeof(u32_E) * WORDS, stream));
                CUDA_CHECK(cudaMemcpyAsync(pq_d, pq, (1 << max_deg >> 1) * sizeof(u32_E) * WORDS, cudaMemcpyHostToDevice, stream));
            }
            CUDA_CHECK(cudaMallocAsync(&param_d, sizeof(mont256::Params), stream));
            CUDA_CHECK(cudaMemcpyAsync(param_d, param, sizeof(mont256::Params), cudaMemcpyHostToDevice, stream));

            if ((config.stage1_mode == SSIP_config::stage1_warp_no_twiddle_no_smem ||
            config.stage1_mode == SSIP_config::stage1_warp_no_twiddle_opt_smem ||
            config.stage1_mode == SSIP_config::stage1_warp_no_twiddle) &&
            (config.stage2_mode == SSIP_config::stage2_warp_no_twiddle_no_share)) {
                CUDA_CHECK(cudaMallocAsync(&roots_d, len / 2 * WORDS * sizeof(u32), stream));
                CUDA_CHECK(cudaMemcpyAsync(roots_d, roots, len / 2 * WORDS * sizeof(u32), cudaMemcpyHostToDevice, stream));
            } else {
                CUDA_CHECK(cudaMallocAsync(&roots_d, len * WORDS * sizeof(u32), stream));
                CUDA_CHECK(cudaMemcpyAsync(roots_d, roots, len * WORDS * sizeof(u32), cudaMemcpyHostToDevice, stream));
            }

            if (inverse) {
                CUDA_CHECK(cudaMallocAsync(&inv_n_d, WORDS * sizeof(u32), stream));
                CUDA_CHECK(cudaMemcpyAsync(inv_n_d, inv_n, WORDS * sizeof(u32), cudaMemcpyHostToDevice, stream));
            }

            if (process) {
                CUDA_CHECK(cudaMallocAsync(&zeta_d, 2 * WORDS * sizeof(u32), stream));
                CUDA_CHECK(cudaMemcpyAsync(zeta_d, zeta, 2 * WORDS * sizeof(u32), cudaMemcpyHostToDevice, stream));
            }

            if (!success) {
                if (pq != nullptr) CUDA_CHECK(cudaFreeAsync(pq_d, stream));
                CUDA_CHECK(cudaFreeAsync(roots_d, stream));
                CUDA_CHECK(cudaFreeAsync(param_d, stream));
            } else {
                this->on_gpu = true;
            }
            return first_err;
        }

        cudaError_t clean_gpu(cudaStream_t stream = 0) override {
            std::unique_lock<std::shared_mutex> wlock(this->mtx);
            if (!this->on_gpu) return cudaSuccess;
            bool success = true;
            cudaError_t first_err = cudaSuccess;

            if (pq != nullptr) CUDA_CHECK(cudaFreeAsync(pq_d, stream));
            CUDA_CHECK(cudaFreeAsync(roots_d, stream));
            CUDA_CHECK(cudaFreeAsync(param_d, stream));

            if (inverse) CUDA_CHECK(cudaFreeAsync(inv_n_d, stream));
            if (process) CUDA_CHECK(cudaFreeAsync(zeta_d, stream));

            this->on_gpu = false;
            return first_err;
        }

        cudaError_t ntt(u32 * data, cudaStream_t stream = 0, u32 start_n = 0, u32 **dev_ptr = nullptr) override {
            bool success = true;
            cudaError_t first_err = cudaSuccess;

            if (log_len == 0) return first_err;

            cudaEvent_t start, end;
            if (success) CUDA_CHECK(cudaEventCreate(&start));
            if (success) CUDA_CHECK(cudaEventCreate(&end));

            std::shared_lock<std::shared_mutex> rlock(this->mtx);
            if (success) {
                while(!this->on_gpu) {
                    rlock.unlock();
                    CUDA_CHECK(to_gpu(stream));
                    rlock.lock();
                }
            }

            this->sem.acquire();

            u32 * x;
            if (success) CUDA_CHECK(cudaMallocAsync(&x, len * WORDS * sizeof(u32), stream));
            if (process && !inverse) {
                if (success) CUDA_CHECK(cudaMemcpyAsync(x, data, start_n * WORDS * sizeof(u32), cudaMemcpyHostToDevice, stream));
            } else {
                if (success) CUDA_CHECK(cudaMemcpyAsync(x, data, len * WORDS * sizeof(u32), cudaMemcpyHostToDevice, stream));
            }
            if (dev_ptr != nullptr) *dev_ptr = x;

            if (debug) {
                dim3 block(768);
                dim3 grid((len - 1) / block.x + 1);
                if (success) number_to_element <WORDS> <<< grid, block, 0, stream >>> (x, len, param_d);
                if (success) CUDA_CHECK(cudaGetLastError());
                if (success) CUDA_CHECK(cudaEventRecord(start));
            }

            int log_stride = log_len - 1;
            constexpr u32 io_group = 1 << (log2_int(WORDS - 1) + 1);
            
            while (log_stride >= log_len / 2) {
                u32 deg = std::min((int)max_deg_stage1, (log_stride + 1 - log_len / 2));

                u32 group_num = std::min((int)(len / (1 << deg)), 1 << (config.max_threads_stage1_log - (deg - 1)));

                u32 block_sz = (1 << (deg - 1)) * group_num;
                assert(block_sz <= (1 << config.max_threads_stage1_log));
                u32 block_num = len / 2 / block_sz;
                assert(block_num * 2 * block_sz == len);

                dim3 block(block_sz);
                dim3 grid(block_num);

                using WarpExchangeT = cub::WarpExchange<u32, io_group, io_group>;

                switch (config.stage1_mode)
                {
                case SSIP_config::stage1_naive: {
                    auto kernel = SSIP_NTT_stage1 <WORDS, io_group>;

                    u32 shared_size = (sizeof(u32) * ((1 << deg) + 1) * WORDS) * group_num;
                    if (config.stage1_coalesced_roots) shared_size += (sizeof(typename WarpExchangeT::TempStorage) * (block.x / io_group));

                    if (success) CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size));

                    if (success) kernel <<< grid, block, shared_size, stream >>>(x, pq_d, log_len, log_stride, deg, max_deg, param_d, 1 << (deg - 1), roots_d, config.stage1_coalesced_roots);
                    if (success) CUDA_CHECK(cudaGetLastError());
                    break;
                }
                case SSIP_config::stage1_warp: {
                    auto kernel = SSIP_NTT_stage1_warp <WORDS, io_group>;

                    u32 shared_size = (sizeof(u32) * ((1 << deg) + 1) * WORDS) * group_num;
                    if (config.stage1_coalesced_roots) shared_size += (sizeof(typename WarpExchangeT::TempStorage) * (block.x / io_group));

                    if (success) CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size));

                    if (success) kernel <<< grid, block, shared_size, stream >>>(x, pq_d, log_len, log_stride, deg, max_deg, param_d, 1 << (deg - 1), roots_d, config.stage1_coalesced_roots);
                    if (success) CUDA_CHECK(cudaGetLastError());
                    break;
                }
                case SSIP_config::stage1_warp_no_twiddle: {
                    if (config.stage1_coalesced_roots) {
                        auto kernel = SSIP_NTT_stage1_warp_no_twiddle_coalesced_roots <WORDS, io_group>;

                        u32 shared_size = (sizeof(typename WarpExchangeT::TempStorage) * (block.x / io_group)) +  (sizeof(u32) * ((1 << deg) + 1) * WORDS) * group_num;
                        
                        if (success) CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size));

                        if (success) kernel <<< grid, block, shared_size, stream >>>(x, log_len, log_stride, deg, param_d, 1 << (deg - 1), roots_d);
                        if (success) CUDA_CHECK(cudaGetLastError());
                    } else {
                        auto kernel = (log_stride == log_len - 1 && (process && (!inverse))) ? 
                        SSIP_NTT_stage1_warp_no_twiddle <WORDS, io_group, true> : SSIP_NTT_stage1_warp_no_twiddle <WORDS, io_group, false>;

                        u32 shared_size = (sizeof(u32) * ((1 << deg) + 1) * WORDS) * group_num;
                        
                        if (success) CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size));

                        if (success) kernel <<< grid, block, shared_size, stream >>>(x, log_len, log_stride, deg, param_d, 1 << (deg - 1), roots_d, zeta_d, start_n);
                        
                        if (success) CUDA_CHECK(cudaGetLastError());
                    }
                    break;
                }
                case SSIP_config::stage1_warp_no_twiddle_no_smem: {
                    auto kernel = SSIP_NTT_stage1_warp_no_smem_no_twiddle <WORDS, io_group>;

                    u32 shared_size = (sizeof(typename WarpExchangeT::TempStorage) * (block.x / io_group));
                    
                    if (success) CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size));

                    if (success) kernel <<< grid, block, shared_size, stream >>>(x, log_len, log_stride, deg, param_d, 1 << (deg - 1), roots_d);
                    if (success) CUDA_CHECK(cudaGetLastError());
                    break;
                }
                case SSIP_config::stage1_warp_no_twiddle_opt_smem: {
                    auto kernel = SSIP_NTT_stage1_warp_no_twiddle_opt_smem <WORDS, io_group>;

                    u32 shared_size = CONVERT_SADDR(sizeof(u32) * ((1 << deg) + 1) * WORDS) * group_num;
                    
                    if (success) CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size));

                    if (success) kernel <<< grid, block, shared_size, stream >>>(x, log_len, log_stride, deg, param_d, 1 << (deg - 1), roots_d);
                    if (success) CUDA_CHECK(cudaGetLastError());
                    break;
                }
                default: {
                    std::cerr << "Invalid stage1_mode" << std::endl;
                    exit(1);
                    break;
                }
                }

                log_stride -= deg;
            }

            assert (log_stride == log_len / 2 - 1);

            while (log_stride >= 0) {
                u32 deg = std::min((int)max_deg_stage2, log_stride + 1);

                using WarpExchangeT = cub::WarpExchange<u32, io_group, io_group>;

                switch (config.stage2_mode)
                {
                case SSIP_config::stage2_naive: {
                    u32 group_num = std::min((int)(len / (1 << (deg << 1))), 1 << (config.max_threads_stage2_log - 2 * (deg - 1)));

                    u32 block_sz = (1 << ((deg - 1) << 1)) * group_num;
                    assert(block_sz <= (1 << config.max_threads_stage2_log));
                    u32 block_num = len / 4 / block_sz;
                    assert(block_num * 4 * block_sz == len);

                    dim3 block(block_sz);
                    dim3 grid(block_num);

                    u32 shared_size = (sizeof(u32) * ((1 << (deg << 1)) + 1) * WORDS) * group_num;
                    if (config.stage2_coalesced_roots) shared_size += (sizeof(typename WarpExchangeT::TempStorage) * (block.x / io_group)); 

                    auto kernel = SSIP_NTT_stage2 <WORDS, io_group>;

                    if (success) CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size));

                    if (success) kernel <<< grid, block, shared_size, stream >>>(x, pq_d, log_len, log_stride, deg, max_deg, param_d, ((1 << (deg << 1)) >> 2), roots_d, config.stage2_coalesced_roots);
                    if (success) CUDA_CHECK(cudaGetLastError());
                    break;
                }
                case SSIP_config::stage2_naive_2_per_thread: {
                    u32 group_num = std::min((int)(len / (1 << (deg << 1))), 1 << (config.max_threads_stage2_log - (2 * deg - 1)));

                    u32 block_sz = (1 << (deg * 2 - 1)) * group_num;
                    assert(block_sz <= (1 << config.max_threads_stage2_log));
                    u32 block_num = len / 2 / block_sz;
                    assert(block_num * 2 * block_sz == len);

                    dim3 block(block_sz);
                    dim3 grid(block_num);

                    u32 shared_size = (sizeof(u32) * ((1 << (deg << 1)) + 1) * WORDS) * group_num;
                    if (config.stage2_coalesced_roots) shared_size += (sizeof(typename WarpExchangeT::TempStorage) * (block.x / io_group)); 

                    auto kernel = SSIP_NTT_stage2_two_per_thread <WORDS, io_group>;

                    if (success) CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size));
                    if (success) kernel <<< grid, block, shared_size, stream >>>(x, pq_d, log_len, log_stride, deg, max_deg, param_d, ((1 << (deg << 1)) >> 1), roots_d, config.stage2_coalesced_roots);
                    if (success) CUDA_CHECK(cudaGetLastError());
                    break;
                }
                case SSIP_config::stage2_warp: {
                    u32 group_num = std::min((int)(len / (1 << (deg << 1))), 1 << (config.max_threads_stage2_log - 2 * (deg - 1)));

                    u32 block_sz = (1 << ((deg - 1) << 1)) * group_num;
                    assert(block_sz <= (1 << config.max_threads_stage2_log));
                    u32 block_num = len / 4 / block_sz;
                    assert(block_num * 4 * block_sz == len);

                    dim3 block(block_sz);
                    dim3 grid(block_num);

                    u32 shared_size = (sizeof(u32) * ((1 << (deg << 1)) + 1) * WORDS) * group_num;
                    if (config.stage2_coalesced_roots) shared_size += (sizeof(typename WarpExchangeT::TempStorage) * (block.x / io_group)); 

                    auto kernel = SSIP_NTT_stage2_warp <WORDS, io_group>;

                    if (success) CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size));

                    if (success) kernel <<< grid, block, shared_size, stream >>>(x, pq_d, log_len, log_stride, deg, max_deg, param_d, ((1 << (deg << 1)) >> 2), roots_d, config.stage2_coalesced_roots);
                    if (success) CUDA_CHECK(cudaGetLastError());
                    break;
                }
                case SSIP_config::stage2_warp_2_per_thread: {
                    u32 group_num = std::min((int)(len / (1 << (deg << 1))), 1 << (config.max_threads_stage2_log - (2 * deg - 1)));

                    u32 block_sz = (1 << (deg * 2 - 1)) * group_num;
                    assert(block_sz <= (1 << config.max_threads_stage2_log));
                    u32 block_num = len / 2 / block_sz;
                    assert(block_num * 2 * block_sz == len);

                    dim3 block(block_sz);
                    dim3 grid(block_num);

                    u32 shared_size = (sizeof(u32) * ((1 << (deg << 1)) + 1) * WORDS) * group_num;
                    if (config.stage2_coalesced_roots) shared_size += (sizeof(typename WarpExchangeT::TempStorage) * (block.x / io_group)); 

                    auto kernel = SSIP_NTT_stage2_two_per_thread_warp <WORDS, io_group>;

                    if (success) CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size));
                    if (success) kernel <<< grid, block, shared_size, stream >>>(x, pq_d, log_len, log_stride, deg, max_deg, param_d, ((1 << (deg << 1)) >> 1), roots_d, config.stage2_coalesced_roots);
                    if (success) CUDA_CHECK(cudaGetLastError());
                    break;
                }
                case SSIP_config::stage2_warp_no_share: {
                    u32 group_num = std::min((int)(len / (1 << (deg << 1))), 1 << (config.max_threads_stage2_log - 2 * (deg - 1)));

                    u32 block_sz = (1 << ((deg - 1) << 1)) * group_num;
                    assert(block_sz <= (1 << config.max_threads_stage2_log));
                    u32 block_num = len / 4 / block_sz;
                    assert(block_num * 4 * block_sz == len);

                    dim3 block(block_sz);
                    dim3 grid(block_num);

                    u32 shared_size = (sizeof(typename WarpExchangeT::TempStorage) * (block.x / io_group)); 

                    auto kernel = SSIP_NTT_stage2_warp_no_share <WORDS, io_group>;

                    if (success) CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size));

                    if (success) kernel <<< grid, block, shared_size, stream >>>(x, pq_d, log_len, log_stride, deg, max_deg, param_d, ((1 << (deg << 1)) >> 2), roots_d, config.stage2_coalesced_roots);
                    if (success) CUDA_CHECK(cudaGetLastError());
                    break;
                }
                case SSIP_config::stage2_warp_2_per_thread_no_share: {
                    u32 group_num = std::min((int)(len / (1 << (deg << 1))), 1 << (config.max_threads_stage2_log - (2 * deg - 1)));

                    u32 block_sz = (1 << (deg * 2 - 1)) * group_num;
                    assert(block_sz <= (1 << config.max_threads_stage2_log));
                    u32 block_num = len / 2 / block_sz;
                    assert(block_num * 2 * block_sz == len);

                    dim3 block(block_sz);
                    dim3 grid(block_num);

                    u32 shared_size = (sizeof(typename WarpExchangeT::TempStorage) * (block.x / io_group));

                    auto kernel = SSIP_NTT_stage2_two_per_thread_warp_no_share <WORDS, io_group>;

                    if (success) CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size));
                    if (success) kernel <<< grid, block, shared_size, stream >>>(x, pq_d, log_len, log_stride, deg, max_deg, param_d, ((1 << (deg << 1)) >> 1), roots_d, config.stage2_coalesced_roots);
                    if (success) CUDA_CHECK(cudaGetLastError());
                    break;
                }
                case SSIP_config::stage2_warp_no_twiddle_no_share: {
                    u32 group_num = std::min((int)(len / (1 << (deg << 1))), 1 << (config.max_threads_stage2_log - 2 * (deg - 1)));

                    u32 block_sz = (1 << ((deg - 1) << 1)) * group_num;
                    assert(block_sz <= (1 << config.max_threads_stage2_log));
                    u32 block_num = len / 4 / block_sz;
                    assert(block_num * 4 * block_sz == len);

                    dim3 block(block_sz);
                    dim3 grid(block_num);

                    u32 shared_size = (sizeof(typename WarpExchangeT::TempStorage) * (block.x / io_group)); 

                    auto kernel = inverse && (log_stride - (int)deg < 0) ?
                    (process ? SSIP_NTT_stage2_warp_no_share_no_twiddle <WORDS, io_group, true, true>
                    : SSIP_NTT_stage2_warp_no_share_no_twiddle <WORDS, io_group, true, false>)
                    : SSIP_NTT_stage2_warp_no_share_no_twiddle <WORDS, io_group, false, false>;

                    if (success) CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size));

                    if (success) kernel <<< grid, block, shared_size, stream >>>(x, log_len, log_stride, deg, param_d, ((1 << (deg << 1)) >> 2), roots_d, inv_n_d, zeta_d);

                    if (success) CUDA_CHECK(cudaGetLastError());
                    break;
                }
                default: {
                    std::cerr << "Invalid stage2_mode" << std::endl;
                    exit(1);
                    break;
                }
                }

                log_stride -= deg;
            }

            if (debug) {
                if (success) CUDA_CHECK(cudaEventRecord(end));
                if (success) CUDA_CHECK(cudaEventSynchronize(end));
                if (success) CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
                dim3 block(768);
                dim3 grid((len - 1) / block.x + 1);
                if (success) element_to_number <WORDS> <<< grid, block, 0, stream >>> (x, len, param_d);
                if (success) CUDA_CHECK(cudaGetLastError());
            }
            
            rlock.unlock();

            if (success && dev_ptr == nullptr) CUDA_CHECK(cudaMemcpyAsync(data, x, len * WORDS * sizeof(u32), cudaMemcpyDeviceToHost, stream));

            if (success && stream == 0) CUDA_CHECK(cudaStreamSynchronize(stream));

            // manually tackle log_len == 1 case because the stage2 kernel won't run
            if (log_len == 1) {
                if (inverse) {
                    auto env = mont256::Env::host_new(*param);
                    env.host_mul(mont256::Element::load(data), mont256::Element::load(inv_n)).store(data);
                    env.host_mul(mont256::Element::load(data + WORDS), mont256::Element::load(inv_n)).store(data + WORDS);
                    if (process) {
                        env.host_mul(mont256::Element::load(data + WORDS), mont256::Element::load(zeta)).store(data + WORDS);
                    }
                }
            }

            if (!(process && (!inverse)))CUDA_CHECK(cudaFreeAsync(x, stream));

            if (debug) CUDA_CHECK(clean_gpu(stream));

            this->sem.release();

            return first_err;
        }
    };
}