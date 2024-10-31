#include <random>
#include <ctime>
#include <cuda_runtime.h>
#include "../src/4step_ntt.cuh"
#include "small_field.cuh"

#define P (3221225473   )
#define root (5)

// // 3221225473
// const auto params = mont256::Params {
//   .m = BIG_INTEGER_CHUNKS8(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xc0000001),
//   .r_mod = BIG_INTEGER_CHUNKS8(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x9fc05273),
//   .r2_mod = BIG_INTEGER_CHUNKS8(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x9c229677),
//   .m_prime = 3221225471
// };

inline unsigned long long qpow(unsigned long long x, unsigned long long y) {
    unsigned long long base = 1ll;
    while(y) {
        if (y & 1ll) base = (base * x) % P;
        x = (x * x) % P;
        y >>= 1ll;
    }
    return base;
}

inline unsigned long long inv(unsigned long long x) {
    return qpow(x, P - 2);
}

void swap(unsigned long long &a, unsigned long long &b) {
    long long tmp = a;
    a = b;
    b = tmp;
}

void ntt_cpu(unsigned long long data[], unsigned long long reverse[], long long len, unsigned long long omega) {

    // rearrange the coefficients
    for (unsigned long long i = 0; i < len; i++) {
        if (i < reverse[i]) swap(data[i], data[reverse[i]]);
    }

    for (unsigned long long stride = 1ll; stride < len; stride <<= 1ll) {
        unsigned long long gap = qpow(omega, (P - 1ll) / (stride << 1ll));
        for (unsigned long long start = 0; start < len; start += (stride << 1ll)) {
            for (unsigned long long offset = 0, w = 1ll; offset < stride; offset++, w = (gap * w) % P) {
                unsigned long long a = data[start + offset], b = w * data[start + offset + stride] % P;
                data[start + offset] = (a + b) % P;
                data[start + offset + stride] = (a - b + P) % P;
            }
        }
    }
}

#define WORDS 8ll

int main() {
    unsigned long long *data, *reverse, *data_copy;
    unsigned long long l,length = 1ll;
    int bits = 0;

    cudaSetDevice(0);

    l = 1 << 24;

    while (length < l) {
        length <<= 1ll;
        bits ++;
    }

    data = new unsigned long long[length];
    data_copy = new unsigned long long[length];
    reverse = new unsigned long long [length];

    reverse[0] = 0;
    for (long long i = 0; i < length; i++) {
        reverse[i] = (reverse[i >> 1ll] >> 1ll) | ((i & 1ll) << (bits - 1ll) ); //reverse the bits
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    for (long long i = 0; i < length; i++) {
        data[i] = i % P;
        data_copy[i] = data[i];
    }

    // cpu implementation
    {
        clock_t start = clock();

        ntt_cpu(data, reverse, length, root);

        clock_t end = clock();
        printf("cpu: %lfms\n",(double)(end - start) / CLOCKS_PER_SEC * 1000);
    }

    uint *src_gpu, *dst_gpu;
    cudaHostAlloc(&src_gpu, length * WORDS * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&dst_gpu, length * WORDS * sizeof(int), cudaHostAllocDefault);


    uint unit[WORDS];
    memset(unit, 0, sizeof(uint) * WORDS);
    unit[0] = root;

    memset(src_gpu, 0, sizeof(uint) * length * WORDS);
    for (int i = 0; i < length; i++) {
        src_gpu[i * WORDS] = data_copy[i];
    }
    
    ntt::offchip_ntt<small_field::Element>(src_gpu, dst_gpu, bits, unit);

    // printf("recompute: %fms\n", rc.milliseconds);

    for (long long i = 0; i < length; i++) {
        if (dst_gpu[i * WORDS] != data[i]) {
            printf("%lld %u %lld\n", data[i], dst_gpu[i * WORDS], i);
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

    cudaFreeHost(dst_gpu);
    cudaFreeHost(src_gpu);
    return 0;
}