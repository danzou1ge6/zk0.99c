#pragma once

#include "../../mont/src/mont.cuh"
#include <cuda_runtime.h>
#include <cassert>
#include <cub/cub.cuh>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scatter.h>
#include <thrust/execution_policy.h>
#include <bit>
#include <cuda/barrier>

namespace NTT {
    typedef uint u32;
    typedef unsigned long long u64;
    typedef uint u32_E;
    typedef uint u32_N;

    __constant__ u32_E omegas_c[32 * 8]; // need to be revised if words changed

    template <u32 WORDS>
    struct element_pack {
        u32 data[WORDS];
    };

    template <u32 WORDS>
    class gen_roots_cub {
        public:
        struct get_iterator_to_range {
            __host__ __device__ __forceinline__ auto operator()(u32 index) {
                return thrust::make_constant_iterator(input_d[index]);
            }
            element_pack<WORDS> *input_d;
        };

        struct get_ptr_to_range {
            __host__ __device__ __forceinline__ auto operator()(u32 index) {
                return output_d + offsets_d[index];
            }
            element_pack<WORDS> *output_d;
            u32 *offsets_d;
        };

        struct get_run_length {
            __host__ __device__ __forceinline__ auto operator()(u32 index) {
                return offsets_d[index + 1] - offsets_d[index];
            }
            uint32_t *offsets_d;
        };
        struct mont_mul {
            __device__ __forceinline__ element_pack<WORDS> operator()(const element_pack<WORDS> &a, const element_pack<WORDS> &b) {
                auto env = mont256::Env(*param_d);
                element_pack<WORDS> res;
                auto a_ = mont256::Element::load(a.data);
                auto b_ = mont256::Element::load(b.data);
                auto c_ = env.mul(a_, b_);
                c_.store(res.data);
                return res;
            }
            mont256::Params* param_d;
        };


        __host__ __forceinline__ void operator() (u32_E * roots, u32 len, mont256::Element &unit, const mont256::Params &param) {
            auto env_host = mont256::Env::host_new(param);

            const u32 num_ranges = 2;

            element_pack<WORDS> input[num_ranges]; // {one, unit}
            env_host.one().store(input[0].data);
            unit.store(input[1].data);
            
            element_pack<WORDS> * input_d;
            cudaMalloc(&input_d, num_ranges * sizeof(element_pack<WORDS>));
            cudaMemcpy(input_d, input, num_ranges * sizeof(element_pack<WORDS>), cudaMemcpyHostToDevice);

            u32 offset[] = {0, 1, len};
            u32 * offset_d;
            cudaMalloc(&offset_d, (num_ranges + 1) * sizeof(u32));
            cudaMemcpy(offset_d, offset, (num_ranges + 1) * sizeof(u32), cudaMemcpyHostToDevice);

            element_pack<WORDS> * output_d;
            cudaMalloc(&output_d, len * sizeof(element_pack<WORDS>));

            // Returns a constant iterator to the element of the i-th run
            thrust::counting_iterator<uint32_t> iota(0);
            auto iterators_in = thrust::make_transform_iterator(iota, get_iterator_to_range{input_d});

            // Returns the run length of the i-th run
            auto sizes = thrust::make_transform_iterator(iota, get_run_length{offset_d});

            // Returns pointers to the output range for each run
            auto ptrs_out = thrust::make_transform_iterator(iota, get_ptr_to_range{output_d, offset_d});

            // Determine temporary device storage requirements
            void *tmp_storage_d = nullptr;
            size_t temp_storage_bytes = 0;

            cub::DeviceCopy::Batched(tmp_storage_d, temp_storage_bytes, iterators_in, ptrs_out, sizes, num_ranges);

            // Allocate temporary storage
            cudaMalloc(&tmp_storage_d, temp_storage_bytes);

            // Run batched copy algorithm (used to perform runlength decoding)
            // output_d       <-- [one, unit, unit, ... , unit]
            cub::DeviceCopy::Batched(tmp_storage_d, temp_storage_bytes, iterators_in, ptrs_out, sizes, num_ranges);

            cudaFree(tmp_storage_d);
            cudaFree(input_d);
            cudaFree(offset_d);
            
            tmp_storage_d = nullptr;
            temp_storage_bytes = 0;

            mont256::Params *param_d;
            cudaMalloc(&param_d, sizeof(mont256::Params));
            cudaMemcpy(param_d, &param, sizeof(mont256::Params), cudaMemcpyHostToDevice);

            auto op = mont_mul{param_d};

            cub::DeviceScan::InclusiveScan(tmp_storage_d, temp_storage_bytes, output_d, op, len);
            cudaMalloc(&tmp_storage_d, temp_storage_bytes);
            cub::DeviceScan::InclusiveScan(tmp_storage_d, temp_storage_bytes, output_d, op, len);

            cudaMemcpy(roots, output_d, len * sizeof(element_pack<WORDS>), cudaMemcpyDeviceToHost);
            cudaFree(output_d);
            cudaFree(tmp_storage_d);
            cudaFree(param_d);
        }
    };

    template<u32 WORDS>
    __global__ void element_to_number(u32* data, u32 len, mont256::Params* param) {
        auto env = mont256::Env(*param);
        u64 index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= len) return;
        env.to_number(mont256::Element::load(data + index * WORDS)).store(data + index * WORDS);
    }

    template<u32 WORDS>
    __global__ void number_to_element(u32* data, u32 len, mont256::Params* param) {
        auto env = mont256::Env(*param);
        u64 index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= len) return;
        env.from_number(mont256::Number::load(data + index * WORDS)).store(data + index * WORDS);
    }
    
    template<u64 WORDS>
    __global__ void rearrange(u32_E * data, u32 log_len) {
        u32 index = blockIdx.x * (blockDim.x / WORDS) + threadIdx.x / WORDS;
        u32 word = threadIdx.x & (WORDS - 1);
        if (index >= 1 << log_len) return;
        u32 rindex = (__brev(index) >> (32 - log_len));
        
        if (rindex >= index) return;

        u32 tmp = data[index * WORDS + word];
        data[index * WORDS + word] = data[rindex * WORDS + word];
        data[rindex * WORDS + word] = tmp;
    }

    template<u32 WORDS>
    __global__ void naive(u32_E* data, u32 len, u32_E* roots, u32 stride, mont256::Params* param) {
        auto env = mont256::Env(*param);

        u32 id = (blockDim.x * blockIdx.x + threadIdx.x);
        if (id << 1 >= len) return;

        u64 offset = id % stride;
        u64 pos = ((id - offset) << 1) + offset;

        auto a = mont256::Element::load(data + (pos * WORDS));
        auto b = mont256::Element::load(data + ((pos + stride) * WORDS));
        auto w = mont256::Element::load(roots + (offset * len / (stride << 1)) * WORDS);

        b = env.mul(b, w);
        
        auto tmp = a;
        a = env.add(a, b);
        b = env.sub(tmp, b);
        
        a.store(data + (pos * WORDS));
        b.store(data + ((pos + stride) * WORDS));
    }


    template<u32 WORDS>
    class naive_ntt {
        const mont256::Params param;
        mont256::Element unit;
        bool debug;
        u32 * reverse;
        u32_E * roots;
        const u32 log_len;
        const u64 len;
        
        void gen_reverse() {
            reverse = new u32[len];
            reverse[0] = 0;
            for (u32 i = 0; i < len; i++) {
                reverse[i] = (reverse[i >> 1] >> 1) | ((i & 1) << (log_len - 1) ); //reverse the bits
            }
        }

        void gen_roots(u32_E * roots, u32 len) {
            auto env = mont256::Env::host_new(param);
            env.one().store(roots);
            auto last_element = mont256::Element::load(roots);
            for (u64 i = WORDS; i < ((u64)len) * WORDS; i+= WORDS) {
                last_element = env.host_mul(last_element, unit);
                last_element.store(roots + i);
            }
        }

        public:
        float milliseconds = 0;

        naive_ntt(const mont256::Params &param, const u32_N* omega, u32 log_len, bool debug) : param(param), log_len(log_len), len(1 << log_len), debug(debug) {
            
            auto env = mont256::Env::host_new(param);
            // unit = qpow(omega, (P - 1ll) / len)
            unit = env.host_from_number(mont256::Number::load(omega));
            auto exponent = mont256::Number::load(param.m);
            auto one = mont256::Number::zero();
            one.c0 = 1;
            exponent = exponent.host_sub(one);
            exponent = exponent.slr(log_len);
            unit = env.host_pow(unit, exponent);

            roots = (u32_E *) malloc(((u64)len) * WORDS * sizeof(u32_E));
            // gen_roots(roots, len);
            gen_roots_cub<WORDS> gen;
            gen(roots, len, unit, param);

            gen_reverse();
        }

        ~naive_ntt() {
            free(roots);
            free(reverse);
        }

        void ntt(u32 * data) {
            cudaEvent_t start, end, start1, end1;
            cudaEventCreate(&start);
            cudaEventCreate(&end);
            cudaEventCreate(&start1);
            cudaEventCreate(&end1);

            u32 * reverse_d;
            cudaMalloc(&reverse_d, len * sizeof(u32));
            cudaMemcpy(reverse_d, reverse, len * sizeof(u32), cudaMemcpyHostToDevice);
            
            u32 * buff1, * buff2;
            cudaMalloc(&buff1, len * WORDS * sizeof(u32));
            cudaMalloc(&buff2, len * WORDS * sizeof(u32));
            
            mont256::Params *param_d;
            cudaMalloc(&param_d, sizeof(mont256::Params));
            cudaMemcpy(param_d, &param, sizeof(mont256::Params), cudaMemcpyHostToDevice);

            // rearrange
            element_pack<WORDS> * input_d = (element_pack<WORDS> *) buff1;
            cudaMemcpy(input_d, data, len * sizeof(element_pack<WORDS>), cudaMemcpyHostToDevice);
            // element_pack<WORDS> * output_d = (element_pack<WORDS> *) buff2;

            if (debug) {
                dim3 block(768);
                dim3 grid((len - 1) / block.x + 1);
                number_to_element <WORDS> <<< grid, block >>> ((u32_E *) buff1, len, param_d);
                cudaEventRecord(start);
            }

            // thrust::scatter(thrust::device, input_d, input_d + len, reverse_d, output_d);
            dim3 block(1024);
            dim3 grid((((u64) len) * WORDS - 1) / block.x + 1);
            rearrange<WORDS> <<<grid, block >>> (buff1, log_len);

            if (debug) {
                cudaEventRecord(end);
            }

            u32_E * data_d = (u32_E *) buff1;
            u32_E * roots_d = (u32_E *) buff2;

            cudaMemcpy(roots_d, roots, ((u64)len) * WORDS * sizeof(u32_E), cudaMemcpyHostToDevice);

            dim3 ntt_block(768);
            dim3 ntt_grid(((len >> 1) - 1) / ntt_block.x + 1);

            if (debug) {
                cudaEventRecord(start1);
            }

            for (u32 stride = 1; stride < len; stride <<= 1) {
                naive <WORDS> <<<ntt_grid, ntt_block>>> (data_d, len, roots_d, stride, param_d);
            }

            if (debug) {
                cudaEventRecord(end1);
                cudaEventSynchronize(end1);
                float t1, t2;
                cudaEventElapsedTime(&t1, start, end);
                cudaEventElapsedTime(&t2, start1, end1);
                printf("%f\n", t1);
                milliseconds = t1 + t2;
                dim3 block(768);
                dim3 grid((len - 1) / block.x + 1);
                element_to_number <WORDS> <<< grid, block >>> (data_d, len, param_d);
            }

            cudaMemcpy(data, data_d, ((u64)len) * WORDS * sizeof(u32), cudaMemcpyDeviceToHost);

            cudaFree(data_d);
            cudaFree(reverse_d);
            cudaFree(roots_d); 
        }
    };

    template <u32 WORDS>
    __forceinline__ __device__ mont256::Element pow_lookup_constant(u32 exponent, mont256::Env &env) {
        auto res = env.one();
        u32 i = 0;
        while(exponent > 0) {
            if (exponent & 1) {
                res = env.mul(res, mont256::Element::load(omegas_c + (i * WORDS)));
            }
            exponent = exponent >> 1;
            i++;
        }
        return res;
    }

    /*
    * FFT algorithm is inspired from: http://www.bealto.com/gpu-fft_group-1.html
    */
    template <u32 WORDS>
    __global__ void bellperson_kernel(u32_E * x, // Source buffer
                        u32_E * y, // Destination buffer
                        const u32_E * pq, // Precalculated twiddle factors
                        u32 n, // Number of elements
                        u32 lgp, // Log2 of `p` (Read more in the link above)
                        u32 deg, // 1=>radix2, 2=>radix4, 3=>radix8, ...
                        u32 max_deg, // Maximum degree supported, according to `pq` and `omegas`
                        mont256::Params* param)
    {

        // There can only be a single dynamic shared memory item, hence cast it to the type we need.
        extern __shared__ u32 u[];

        auto env = mont256::Env(*param);

        u32 lid = threadIdx.x;//GET_LOCAL_ID();
        u32 lsize = blockDim.x;//GET_LOCAL_SIZE();
        u32 index = blockIdx.x;//GET_GROUP_ID();
        u32 t = n >> deg;
        u32 p = 1 << lgp;
        u32 k = index & (p - 1);

        x += index * WORDS;
        y += (((index - k) << deg) + k) * WORDS;

        u32 count = 1 << deg; // 2^deg
        u32 counth = count >> 1; // Half of count

        u32 counts = count / lsize * lid;
        u32 counte = counts + count / lsize;

        // Compute powers of twiddle
        auto twiddle = pow_lookup_constant<WORDS>((n >> lgp >> deg) * k, env);

        auto tmp = env.pow(twiddle, counts, deg);

        for(u64 i = counts; i < counte; i++) {
            auto num = mont256::Element::load(x + (i * t * WORDS));
            num = env.mul(num, tmp);
            num.store(u + (i * WORDS));
            tmp = env.mul(tmp, twiddle);
        }

        __syncthreads();

        const u32 pqshift = max_deg - deg;
        for(u32 rnd = 0; rnd < deg; rnd++) {
            const u32 bit = counth >> rnd;
            for(u32 i = counts >> 1; i < counte >> 1; i++) {
                const u32 di = i & (bit - 1);
                const u32 i0 = (i << 1) - di;
                const u32 i1 = i0 + bit;
                mont256::Element a, b, tmp, w;
                a = mont256::Element::load(u + (i0 * WORDS));
                b = mont256::Element::load(u + (i1 * WORDS));
                tmp = a;
                a = env.add(a, b);
                b = env.sub(tmp, b);
                if (di != 0) {
                    w = mont256::Element::load(pq + ((di << rnd << pqshift) * WORDS));
                    b = env.mul(b, w);
                }
                a.store(u + (i0 * WORDS));
                b.store(u + (i1 * WORDS));
            }
            __syncthreads();
        }
        

        for(u32 i = counts >> 1; i < counte >> 1; i++) {
            #pragma unroll
            for (u32 j = 0; j < WORDS; j++) {
                y[(((u64)i) * p) * WORDS + j] = u[(__brev(i) >> (32 - deg)) * WORDS + j];
                y[(((u64)(i + counth)) * p) * WORDS + j] = u[(__brev(i + counth) >> (32 - deg)) * WORDS + j];
            }
        }
    }
    
    template<u32 WORDS>
    class bellperson_ntt {
        const u32 max_deg;
        const u32 log_len;
        const u64 len;
        mont256::Params param;
        mont256::Element unit;
        bool debug;
        u32_E *pq; // Precalculated values for radix degrees up to `max_deg`
        u32_E *omegas; // Precalculated values for [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]


        public:
        float milliseconds = 0;
        bellperson_ntt(const mont256::Params param, const u32* omega, u32 log_len, bool debug) : param(param), log_len(log_len), max_deg(std::min(8u, log_len)), len(1 << log_len), debug(debug) {
            // Precalculate:
            auto env = mont256::Env::host_new(param);

            // unit = qpow(omega, (P - 1ll) / len)
            unit = env.host_from_number(mont256::Number::load(omega));
            auto exponent = mont256::Number::load(param.m);
            auto one = mont256::Number::zero();
            one.c0 = 1;
            exponent = exponent.host_sub(one);
            exponent = exponent.slr(log_len);
            unit = env.host_pow(unit, exponent);

            // pq: [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]

            pq = (u32 *) malloc((1 << max_deg >> 1) * sizeof(u32) * WORDS);
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

            // omegas: [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]

            omegas = (u32 *) malloc(32 * sizeof(u32) * WORDS);
            unit.store(omegas);
            auto last = unit;
            for (u32 i = 1; i < 32; i++) {
                last = env.host_square(last);
                last.store(omegas + i * WORDS);
            }
        }

        ~bellperson_ntt() {
            free(pq);
            free(omegas);
        }

        void ntt(u32 * data) {
            cudaEvent_t start, end;
            cudaEventCreate(&start);
            cudaEventCreate(&end);
         
            u32_E* pq_d;

            cudaMalloc(&pq_d, (1 << max_deg >> 1) * sizeof(u32_E) * WORDS);
            cudaMemcpy(pq_d, pq, (1 << max_deg >> 1) * sizeof(u32_E) * WORDS, cudaMemcpyHostToDevice);

            cudaMemcpyToSymbol(omegas_c, omegas, 32 * sizeof(u32_E) * WORDS);

            u32 * x, * y;
            cudaMalloc(&y, len * WORDS * sizeof(u32));
            cudaMalloc(&x, len * WORDS * sizeof(u32));
            cudaMemcpy(x, data, len * WORDS * sizeof(u32), cudaMemcpyHostToDevice);

            mont256::Params* param_d;
            cudaMalloc(&param_d, sizeof(mont256::Params));
            cudaMemcpy(param_d, &param, sizeof(mont256::Params), cudaMemcpyHostToDevice);

            // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
            u32 log_p = 0u;
            
            if (debug) {
                dim3 block(768);
                dim3 grid((len - 1) / block.x + 1);
                number_to_element <WORDS> <<< grid, block >>> (x, len, param_d);
                cudaEventRecord(start);
            }

            // Each iteration performs a FFT round
            while (log_p < log_len) {

                // 1=>radix2, 2=>radix4, 3=>radix8, ...
                u32 deg = std::min(max_deg, log_len - log_p);

                assert(deg - 1 <= 10);

                dim3 block(1 << (deg - 1));
                dim3 grid(len >> deg);

                uint shared_size = (1 << deg) * WORDS * sizeof(u32);

                auto kernel = bellperson_kernel<WORDS>;
            
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size);

                kernel <<< grid, block, shared_size >>> (x, y, pq_d, len, log_p, deg, max_deg, param_d);
                
                auto err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("Error: %s\n", cudaGetErrorString(err));
                }
                log_p += deg;

                u32 * tmp = x;
                x = y;
                y = tmp;
            }

            if (debug) {
                cudaEventRecord(end);
                cudaEventSynchronize(end);
                cudaEventElapsedTime(&milliseconds, start, end);
                dim3 block(768);
                dim3 grid((len - 1) / block.x + 1);
                element_to_number <WORDS> <<< grid, block >>> (x, len, param_d);
            }

            cudaMemcpy(data, x, len * WORDS * sizeof(u32), cudaMemcpyDeviceToHost);

            cudaFree(pq_d);
            cudaFree(y);
            cudaFree(x);

        }
    };

    template <u32 WORDS, u32 io_group>
    __global__ void SSIP_NTT_stage1 (u32_E * x, // input data, shape: [1, len*words] data stored in row major i.e. a_0 ... a_7 b_0 ... b_7 ...
                                    const u32_E * pq, // twiddle factors for shard memory NTT
                                    u32 len, // input data length
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


        constexpr int warp_threads = io_group;
        constexpr int items_per_thread = io_group;
        const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
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
        u64 twiddle = (len >> (log_stride - deg + 1) >> deg) * k;

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
                    u32 gpos = group_id << (lgp + 1);
                    x[gpos * WORDS + io] = u[(i << 1) + io * shared_read_stride];
                    x[(gpos + end_stride) * WORDS + io] = u[(i << 1) + 1 + io * shared_read_stride];
                }
            }
        }
    }

    template <u32 WORDS, u32 io_group>
    __global__ void SSIP_NTT_stage1_warp (u32_E * x, // input data, shape: [1, len*words] data stored in row major i.e. a_0 ... a_7 b_0 ... b_7 ...
                                    const u32_E * pq, // twiddle factors for shard memory NTT
                                    u32 len, // input data length
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


        constexpr int warp_threads = io_group;
        constexpr int items_per_thread = io_group;
        const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
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
        u64 twiddle = (len >> (log_stride - deg + 1) >> deg) * k;

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
                    u32 gpos = group_id << (lgp + 1);
                    x[gpos * WORDS + io] = u[(i << 1) + io * shared_read_stride];
                    x[(gpos + end_stride) * WORDS + io] = u[(i << 1) + 1 + io * shared_read_stride];
                }
            }
        }
    }

    template <u32 WORDS, u32 io_group>
    __global__ void SSIP_NTT_stage2_two_per_thread (u32_E * data, const u32_E * pq, u32 log_len, u32 log_stride, u32 deg, u32 max_deg, mont256::Params* param, u32 group_sz, u32 * roots, bool coalesced_roots) {
        extern __shared__ u32_E s[];

        constexpr int warp_threads = io_group;
        constexpr int items_per_thread = io_group;
        const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
        u32 thread_data[io_group];

        // Specialize WarpExchange for a virtual warp of a threads owning b integer items each
        using WarpExchangeT = cub::WarpExchange<u32, items_per_thread, warp_threads>;

        // Allocate shared memory for WarpExchange
        typename WarpExchangeT::TempStorage *temp_storage = (typename WarpExchangeT::TempStorage*) s;

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
        auto u = s + (sizeof(WarpExchangeT::TempStorage) / sizeof(u32) * (blockDim.x / warp_threads)) + group_id * ((1 << (deg << 1)) + 1) * WORDS;

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

        u64 twiddle = (n >> (log_stride - deg + 1) >> deg) * k;

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

        constexpr int warp_threads = io_group;
        constexpr int items_per_thread = io_group;
        const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
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
        auto u = s + (sizeof(WarpExchangeT::TempStorage) / sizeof(u32) * (blockDim.x / warp_threads)) + group_id * ((1 << (deg << 1)) + 1) * WORDS;

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

        u64 twiddle = (n >> (log_stride - deg + 1) >> deg) * k;

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

        constexpr int warp_threads = io_group;
        constexpr int items_per_thread = io_group;
        const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
        u32 thread_data[io_group];

        // Specialize WarpExchange for a virtual warp of a threads owning b integer items each
        using WarpExchangeT = cub::WarpExchange<u32, items_per_thread, warp_threads>;

        // Allocate shared memory for WarpExchange
        typename WarpExchangeT::TempStorage *temp_storage = (typename WarpExchangeT::TempStorage*) s;

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
        auto u = s + (sizeof(WarpExchangeT::TempStorage) / sizeof(u32) * (blockDim.x / warp_threads)) + group_id * ((1 << (deg << 1)) + 1) * WORDS;

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

        u64 twiddle = (n >> (log_stride - deg + 1) >> deg) * k;

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

        constexpr int warp_threads = io_group;
        constexpr int items_per_thread = io_group;
        const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
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
        auto u = s + (sizeof(WarpExchangeT::TempStorage) / sizeof(u32) * (blockDim.x / warp_threads)) + group_id * ((1 << (deg << 1)) + 1) * WORDS;

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

        u64 twiddle = (n >> (log_stride - deg + 1) >> deg) * k;

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

        u64 twiddle = (n >> (log_stride - deg + 1) >> deg) * k;

        mont256::Element t1;
        if (coalesced_roots && cur_io_group == io_group) {

        } else {
            u32 r1 = __brev((lid << 1) + second_half * lsize) >> (32 - (deg << 1));

            u64 pos = twiddle * (r1 >> deg);
            t1 = mont256::Element::load(roots + pos * WORDS);
        }

        a = env.mul(a, t1);

        if (coalesced_roots && cur_io_group == io_group) {
           
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
        u64 twiddle = (n >> (log_stride - deg + 1) >> deg) * k;

        mont256::Element t1;
        if (coalesced_roots && cur_io_group == io_group) {
            // for (int i = lid_start; i < lid_start + cur_io_group; i++) {
            //     if (io_id < WORDS) {
            //         u32 r = __brev((i << 1)) >> (32 - (deg << 1));
            //         u64 pos = twiddle * (r >> deg);
            //         thread_data[i - lid_start] = roots[pos * WORDS + io_id];
            //     }
            // }

            // // Collectively exchange data into a blocked arrangement across threads
            // WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);
            // t1 = mont256::Element::load(thread_data);
            // __syncwarp();
        } else {
            u32 r1 = __brev((lid << 1)) >> (32 - (deg << 1));

            u64 pos = twiddle * (r1 >> deg);
            t1 = mont256::Element::load(roots + pos * WORDS);
        }

        a = env.mul(a, t1);

        if (coalesced_roots && cur_io_group == io_group) {
        //    for (int i = lid_start; i < lid_start + cur_io_group; i++) {
        //         if (io_id < WORDS) {
        //             u32 r = __brev((i << 1) + 1) >> (32 - (deg << 1));
        //             u64 pos = twiddle * (r >> deg);
        //             thread_data[i - lid_start] = roots[pos * WORDS + io_id];
        //         }
        //     }

        //     // Collectively exchange data into a blocked arrangement across threads
        //     WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);
        //     t1 = mont256::Element::load(thread_data);
        //     __syncwarp();
        } else {
            u32 r2 = __brev((lid << 1) + 1) >> (32 - (deg << 1));

            u64 pos = twiddle * (r2 >> deg);
            t1 = mont256::Element::load(roots + pos * WORDS);
        }

        b = env.mul(b, t1);

        if (coalesced_roots && cur_io_group == io_group) {
        //    for (int i = lid_start; i < lid_start + cur_io_group; i++) {
        //         if (io_id < WORDS) {
        //             u32 r = __brev((i << 1) + lsize * 2) >> (32 - (deg << 1));
        //             u64 pos = twiddle * (r >> deg);
        //             thread_data[i - lid_start] = roots[pos * WORDS + io_id];
        //         }
        //     }

        //     // Collectively exchange data into a blocked arrangement across threads
        //     WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);
        //     t1 = mont256::Element::load(thread_data);
        //     __syncwarp();
        } else {
            u32 r3 = __brev((lid << 1) + (lsize * 2)) >> (32 - (deg << 1));

            u64 pos = twiddle * (r3 >> deg);
            t1 = mont256::Element::load(roots + pos * WORDS);
        }

        c = env.mul(c, t1);

        if (coalesced_roots && cur_io_group == io_group) {
        //    for (int i = lid_start; i < lid_start + cur_io_group; i++) {
        //         if (io_id < WORDS) {
        //             u32 r = __brev((i << 1) + 1 + lsize * 2) >> (32 - (deg << 1));
        //             u64 pos = twiddle * (r >> deg);
        //             thread_data[i - lid_start] = roots[pos * WORDS + io_id];
        //         }
        //     }

        //     // Collectively exchange data into a blocked arrangement across threads
        //     WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);
        //     t1 = mont256::Element::load(thread_data);
        //     __syncwarp();
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

    template <u32 WORDS>
    class self_sort_in_place_ntt {
        const u32 max_threads_stage1_log = 8;
        const u32 max_threads_stage2_log = 8;
        u32 max_deg_stage1;
        u32 max_deg_stage2;
        u32 max_deg;
        const int log_len;
        const u64 len;
        mont256::Params param;
        mont256::Element unit;
        bool debug;
        u32_E *pq; // Precalculated values for radix degrees up to `max_deg`
        u32_E *roots;

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
        float milliseconds = 0;
        self_sort_in_place_ntt(const mont256::Params param, const u32* omega, u32 log_len, bool debug) 
        : param(param), log_len(log_len), len(1 << log_len), debug(debug) {
            u32 deg_stage1 = (log_len + 1) / 2;
            u32 deg_stage2 = log_len / 2;
            max_deg_stage1 = get_deg(deg_stage1, max_threads_stage1_log + 1);
            max_deg_stage2 = get_deg(deg_stage2, (max_threads_stage2_log + 2) / 2); // 4 elements per thread
            // max_deg_stage2 = get_deg(deg_stage2, (max_threads_stage2_log + 1) / 2);
            max_deg = std::max(max_deg_stage1, max_deg_stage2);

            // Precalculate:
            auto env = mont256::Env::host_new(param);

            // unit = qpow(omega, (P - 1ll) / len)
            unit = env.host_from_number(mont256::Number::load(omega));
            auto exponent = mont256::Number::load(param.m);
            auto one = mont256::Number::zero();
            one.c0 = 1;
            exponent = exponent.host_sub(one);
            exponent = exponent.slr(log_len);
            unit = env.host_pow(unit, exponent);

            // pq: [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]

            pq = (u32 *) malloc((1 << max_deg >> 1) * sizeof(u32) * WORDS);
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

            roots = (u32_E *) malloc(((u64)len) * WORDS * sizeof(u32_E));
            gen_roots_cub<WORDS> gen;
            gen(roots, len, unit, param);
        }

        ~self_sort_in_place_ntt() {
            free(pq);
            free(roots);
        }

        void ntt(u32 * data) {
            cudaEvent_t start, end;
            cudaEventCreate(&start);
            cudaEventCreate(&end);
         
            u32_E* pq_d;

            cudaMalloc(&pq_d, (1 << max_deg >> 1) * sizeof(u32_E) * WORDS);
            cudaMemcpy(pq_d, pq, (1 << max_deg >> 1) * sizeof(u32_E) * WORDS, cudaMemcpyHostToDevice);

            u32 * x;
            cudaMalloc(&x, len * WORDS * sizeof(u32));
            cudaMemcpy(x, data, len * WORDS * sizeof(u32), cudaMemcpyHostToDevice);

            mont256::Params* param_d;
            cudaMalloc(&param_d, sizeof(mont256::Params));
            cudaMemcpy(param_d, &param, sizeof(mont256::Params), cudaMemcpyHostToDevice);

            u32 * roots_d;
            cudaMalloc(&roots_d, len * WORDS * sizeof(u32));
            cudaMemcpy(roots_d, roots, len * WORDS * sizeof(u32), cudaMemcpyHostToDevice);

            if (debug) {
                dim3 block(768);
                dim3 grid((len - 1) / block.x + 1);
                number_to_element <WORDS> <<< grid, block >>> (x, len, param_d);
                cudaEventRecord(start);
            }

            int log_stride = log_len - 1;
            constexpr u32 io_group = 1 << (log2_int(WORDS - 1) + 1);
            
            while (log_stride >= log_len / 2) {
                u32 deg = std::min((int)max_deg_stage1, (log_stride + 1 - log_len / 2));

                u32 group_num = std::min((int)(len / (1 << deg)), 1 << (max_threads_stage1_log - (deg - 1)));

                u32 block_sz = (1 << (deg - 1)) * group_num;
                assert(block_sz <= (1 << max_threads_stage1_log));
                u32 block_num = len / 2 / block_sz;
                assert(block_num * 2 * block_sz == len);

                dim3 block(block_sz);
                dim3 grid(block_num);

                using WarpExchangeT = cub::WarpExchange<u32, io_group, io_group>;

                u32 shared_size = (sizeof(typename WarpExchangeT::TempStorage) * (block.x / io_group)) + (sizeof(u32) * ((1 << deg) + 1) * WORDS) * group_num;

                auto kernel = SSIP_NTT_stage1_warp <WORDS, io_group>;
            
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size);

                kernel <<< grid, block, shared_size >>>(x, pq_d, len, log_stride, deg, max_deg, param_d, 1 << (deg - 1), roots_d, true);

                log_stride -= deg;
            }

            assert (log_stride == log_len / 2 - 1);

            while (log_stride >= 0) {
                u32 deg = std::min((int)max_deg_stage2, log_stride + 1);

                // u32 group_num = std::min((int)(len / (1 << (deg << 1))), 1 << (max_threads_stage2_log - (2 * deg - 1)));

                // u32 block_sz = (1 << (deg * 2 - 1)) * group_num;
                // assert(block_sz <= (1 << max_threads_stage2_log));
                // u32 block_num = len / 2 / block_sz;
                // assert(block_num * 2 * block_sz == len);

                u32 group_num = std::min((int)(len / (1 << (deg << 1))), 1 << (max_threads_stage2_log - 2 * (deg - 1)));

                u32 block_sz = (1 << ((deg - 1) << 1)) * group_num;
                assert(block_sz <= (1 << max_threads_stage2_log));
                u32 block_num = len / 4 / block_sz;
                assert(block_num * 4 * block_sz == len);

                dim3 block1(block_sz);
                dim3 grid1(block_num);

                using WarpExchangeT = cub::WarpExchange<u32, io_group, io_group>;

                u32 shared_size = (sizeof(typename WarpExchangeT::TempStorage) * (block1.x / io_group)) + (sizeof(u32) * ((1 << (deg << 1)) + 1) * WORDS) * group_num;

                auto kernel = SSIP_NTT_stage2_warp_no_share <WORDS, io_group>;

                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size);

                kernel <<< grid1, block1, shared_size >>>(x, pq_d, log_len, log_stride, deg, max_deg, param_d, ((1 << (deg << 1)) >> 2), roots_d, false);
                // kernel <<< grid1, block1, shared_size >>>(x, pq_d, log_len, log_stride, deg, max_deg, param_d, ((1 << (deg << 1)) >> 1), roots_d, false);

                log_stride -= deg;
            }

            if (debug) {
                cudaEventRecord(end);
                cudaEventSynchronize(end);
                cudaEventElapsedTime(&milliseconds, start, end);
                dim3 block(768);
                dim3 grid((len - 1) / block.x + 1);
                element_to_number <WORDS> <<< grid, block >>> (x, len, param_d);
            }

            cudaMemcpy(data, x, len * WORDS * sizeof(u32), cudaMemcpyDeviceToHost);

            cudaFree(pq_d);
            cudaFree(x);
            cudaFree(roots_d);

        }
    };
}