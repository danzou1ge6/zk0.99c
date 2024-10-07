#include <random>
#include <ctime>
#include "../src/naive_ntt.cuh"
#include "../src/bellperson_ntt.cuh"
#include "../src/self_sort_in_place_ntt.cuh"
#include <thread>

#define P (3221225473   )
#define root (5)

// 3221225473
const auto params = mont256::Params {
  .m = BIG_INTEGER_CHUNKS8(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xc0000001),
  .r_mod = BIG_INTEGER_CHUNKS8(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x9fc05273),
  .r2_mod = BIG_INTEGER_CHUNKS8(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x9c229677),
  .m_prime = 3221225471
};

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

#define NUM_THREADS 100

int main() {
    unsigned long long *data, *reverse, *data_copy;
    unsigned long long l,length = 1ll;
    int bits = 0;

    cudaSetDevice(0);

    l = 1 << 21;

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

    uint *data_gpu[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        cudaHostAlloc(&data_gpu[i], sizeof(uint) * length * WORDS, cudaHostAllocDefault);
        // data_gpu[i] = new uint [length * WORDS];
    }

    uint unit[WORDS];
    memset(unit, 0, sizeof(uint) * WORDS);
    unit[0] = root;

    for (int i = 0; i < NUM_THREADS; i++) {
        memset(data_gpu[i], 0, sizeof(uint) * length * WORDS);
        for (int j = 0; j < length; j++) {
            data_gpu[i][j * WORDS] = data_copy[j];
        }
    }
    
    cudaStream_t stream[NUM_THREADS];
    uint ***dev_ptr;
    dev_ptr = new uint **[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        cudaStreamCreate(&stream[i]);
    }
    ntt::self_sort_in_place_ntt<WORDS> SSIP(params, unit, bits, true, 100);
    SSIP.to_gpu();

    // for (int i = 0; i < NUM_THREADS; i++) {
    //     SSIP.ntt(data_gpu[i], stream[i]);
    //     printf("dispatch %d\n", i);
    // }
    // printf("end dispatch\n");
    // cudaDeviceSynchronize();
    // printf("end ntt\n");

    std::thread threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i] = std::thread([i, &data_gpu, &stream, &SSIP](){
            SSIP.ntt(data_gpu[i], stream[i]);
        });
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
    cudaDeviceSynchronize();


    for (int i = 0; i < NUM_THREADS; i++) {
        cudaStreamDestroy(stream[i]);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        for (int j = 0; j < length; j++) {
            if (data[j] != data_gpu[i][j * WORDS]) {
                printf("%lld %u %d\n", data[i], data_gpu[i][j * WORDS], j);
                break;
            }
        }
    }

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    delete [] data;
    delete [] data_copy;
    delete [] reverse;
    // for (int i = 0; i < NUM_THREADS; i++) {
    //     delete [] data_gpu[i];
    // }
    delete [] dev_ptr;
    return 0;
}