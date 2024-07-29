#include <iostream>
#include <random>
#include <cassert>
#include <cuda_runtime.h>
#include <ctime>
#include "../src/NTT.cuh"

#define P (469762049 ) // 29 * 2^57 + 1ll
#define root (3)

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

#define WORDS 8

int main() {
    long long *data, *reverse, *data_copy;
    long long l,length = 1ll;
    int bits = 0;

    //scanf("%lld", &l);
    l = qpow(2, 26);

    while (length < l) {
        length <<= 1ll;
        bits ++;
    }


    data = new long long[length];
    data_copy = new long long[length];
    reverse = new long long [length];

    for (long long i = 0; i < length; i++) {
        reverse[i] = (reverse[i >> 1ll] >> 1ll) | ((i & 1ll) << (bits - 1ll) ); //reverse the bits
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    for (long long i = 0; i < length; i++) {
        data[i] = i; std::abs((long long)gen()) % P;
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
    memset(data_gpu, 0, sizeof(*data_gpu) * length);
    for (int i = 0; i < length; i++) {
        data_gpu[i * WORDS] = data[i];
    }

    // TODO: Params
    auto param = mont256::Params();
    uint unit[WORDS];
    memset(unit, 0, sizeof(uint) * WORDS);
    unit[0] = root;
    // naive gpu approach
    
    NTT::naive_ntt<WORDS> naive(param, unit, bits, true);
    naive.ntt(data_gpu);
    printf("naive: %fms\n", naive.milliseconds);

    for (long long i = 0; i < length; i++) {
        if (data[i] != data_gpu[i * WORDS]) {
            printf("%lld %lld %lld\n", data[i], data_copy[i], i);
        }
    }


    delete [] data;
    delete [] data_copy;
    delete [] reverse;
    delete [] data_gpu;

    return 0;
}