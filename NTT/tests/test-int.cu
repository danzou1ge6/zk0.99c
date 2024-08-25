#include <iostream>
#include <random>
#include <cassert>
#include <ctime>
#include "../src/NTT.cuh"

#define P (469762049 )
#define root (3)

// 469762049
const auto params = mont256::Params {
  .m = BIG_INTEGER_CHUNKS8(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1c000001),
  .r_mod = BIG_INTEGER_CHUNKS8(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xea6f185),
  .r2_mod = BIG_INTEGER_CHUNKS8(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x4acda38),
  .m_prime = 469762047
};

inline long long qpow(long long x, long long y) {
    long long base = 1ll;
    while(y) {
        if (y & 1ll) base = (base * x) % P;
        x = (x * x) % P;
        y >>= 1ll;
    }
    return base;
}

inline long long inv(long long x) {
    return qpow(x, P - 2);
}

void swap(long long &a, long long &b) {
    long long tmp = a;
    a = b;
    b = tmp;
}

void ntt_cpu(long long data[], long long reverse[], long long len, long long omega) {

    // rearrange the coefficients
    for (long long i = 0; i < len; i++) {
        if (i < reverse[i]) swap(data[i], data[reverse[i]]);
    }

    for (long long stride = 1ll; stride < len; stride <<= 1ll) {
        long long gap = qpow(omega, (P - 1ll) / (stride << 1ll));
        for (long long start = 0; start < len; start += (stride << 1ll)) {
            for (long long offset = 0, w = 1ll; offset < stride; offset++, w = (gap * w) % P) {
                long long a = data[start + offset], b = w * data[start + offset + stride] % P;
                data[start + offset] = (a + b) % P;
                data[start + offset + stride] = (a - b + P) % P;
                // printf("%lld %lld\n", w, offset);
            }
        }
    }
}

// template <int WORDS>
// void tmp(uint * data, long long * reverse, uint len) {
//     using namespace NTT;
//     long long * reverse_d;
//     cudaMalloc(&reverse_d, len * sizeof(long long));
//     cudaMemcpy(reverse_d, reverse, len * sizeof(long long), cudaMemcpyHostToDevice);

//     u32 * buff1, * buff2;
//     cudaMalloc(&buff1, len * WORDS * sizeof(u32));
//     cudaMalloc(&buff2, len * WORDS * sizeof(u32));

//     element_pack<WORDS> * input_d = (element_pack<WORDS> *) buff1;
//     cudaMemcpy(input_d, data, len * sizeof(element_pack<WORDS>), cudaMemcpyHostToDevice);

//     element_pack<WORDS> * output_d = (element_pack<WORDS> *) buff2;

//     cudaEvent_t start_re, end_re;
//     cudaEventCreate(&start_re);
//     cudaEventCreate(&end_re);
//     cudaEventRecord(start_re);
            
//     thrust::scatter(thrust::device, input_d, input_d + len, reverse_d, output_d);

//     cudaEventRecord(end_re);
//     cudaEventSynchronize(end_re);
//     float milliseconds_re = 0;
//     cudaEventElapsedTime(&milliseconds_re, start_re, end_re);
//     printf("Rearrange_thrust: %f\n", milliseconds_re);

//     cudaMemcpy(data, output_d, len * sizeof(element_pack<WORDS>), cudaMemcpyDeviceToHost);

//     // for (int i = 0; i < len; i ++) {
//     //     printf("%u ", data[i * WORDS]);
//     // }
//     // printf("\n");

//     cudaFree(reverse_d);
//     cudaFree(buff1);
//     cudaFree(buff2);
// }

#define WORDS 8

int main() {
    long long *data, *reverse, *data_copy;
    long long l,length = 1ll;
    int bits = 0;

    cudaSetDevice(0);

    l = 1 << 24;

    while (length < l) {
        length <<= 1ll;
        bits ++;
    }

    data = new long long[length];
    data_copy = new long long[length];
    reverse = new long long [length];

    reverse[0] = 0;
    for (long long i = 0; i < length; i++) {
        reverse[i] = (reverse[i >> 1ll] >> 1ll) | ((i & 1ll) << (bits - 1ll) ); //reverse the bits
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    for (long long i = 0; i < length; i++) {
        data[i] = i % P;std::abs((long long)gen()) % P;
        data_copy[i] = data[i];
    }

    // cpu implementation
    {
        clock_t start = clock();

        ntt_cpu(data, reverse, length, root);

        clock_t end = clock();
        printf("cpu: %lfms\n",(double)(end - start) / CLOCKS_PER_SEC * 1000);
    }

    uint *data_gpu;
    data_gpu = new uint [length * WORDS];

    uint unit[WORDS];
    memset(unit, 0, sizeof(uint) * WORDS);
    unit[0] = root;

    // naive gpu approach
    memset(data_gpu, 0, sizeof(uint) * length * WORDS);
    for (int i = 0; i < length; i++) {
        data_gpu[i * WORDS] = data_copy[i];
    }

    NTT::naive_ntt<WORDS> naive(params, unit, bits, true);
    naive.ntt(data_gpu);
    printf("naive: %fms\n", naive.milliseconds);

    for (long long i = 0; i < length; i++) {
        if (data[i] != data_gpu[i * WORDS]) {
            printf("%lld %u %lld\n", data[i], data_gpu[i * WORDS], i);
            break;
        }
    }

    // bellperson approach
    memset(data_gpu, 0, sizeof(uint) * length * WORDS);
    for (int i = 0; i < length; i++) {
        data_gpu[i * WORDS] = data_copy[i];
    }
    NTT::bellperson_ntt<WORDS> bellperson(params, unit, bits, true);
    bellperson.ntt(data_gpu);
    printf("bellperson: %fms\n", bellperson.milliseconds);

    for (long long i = 0; i < length; i++) {
        if (data[i] != data_gpu[i * WORDS]) {
            printf("%lld %u %lld\n", data[i], data_gpu[i * WORDS], i);
            break;
        }
    }

    // self sort in place approach
    memset(data_gpu, 0, sizeof(uint) * length * WORDS);
    for (int i = 0; i < length; i++) {
        data_gpu[i * WORDS] = data_copy[i];
    }
    NTT::self_sort_in_place_ntt<WORDS> SSIP(params, unit, bits, true);
    SSIP.ntt(data_gpu);
    printf("SSIP: %fms\n", SSIP.milliseconds);

    for (long long i = 0; i < length; i++) {
        if (data[i] != data_gpu[i * WORDS]) {
            printf("%lld %u %lld\n", data[i], data_gpu[i * WORDS], i);
            break;
        }
    }

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    delete [] data;
    delete [] data_copy;
    delete [] reverse;
    delete [] data_gpu;

    return 0;
}