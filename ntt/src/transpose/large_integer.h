#pragma once
#include <cstdint>
#include <cstring>
#include <immintrin.h>

// 大整数表示，大小为N字节(N必须是32的倍数)
template <size_t N>
struct alignas(32) LargeInteger {
    static_assert(N % 32 == 0, "Size must be a multiple of 32 bytes");
    uint8_t data[N];
};

// 帮助类用于AVX内存操作
template <size_t N>
struct AVXHelper {};

// 针对32字节的特化
template <>
struct AVXHelper<32> {
    static inline void copy(const uint8_t* src, uint8_t* dst) {
        __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(src));
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst), v);
    }
};

// 针对64字节的特化
template <>
struct AVXHelper<64> {
    static inline void copy(const uint8_t* src, uint8_t* dst) {
        __m256i v1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src));
        __m256i v2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src + 32));
        
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst), v1);
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst + 32), v2);
    }
};

// 针对96字节的特化
template <>
struct AVXHelper<96> {
    static inline void copy(const uint8_t* src, uint8_t* dst) {
        __m256i v1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src));
        __m256i v2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src + 32));
        __m256i v3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src + 64));
        
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst), v1);
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst + 32), v2);
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst + 64), v3);
    }
};