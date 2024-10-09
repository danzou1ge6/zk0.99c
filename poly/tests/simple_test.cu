#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "../src/poly.cuh"
#include "../../mont/src/bn256_fr.cuh"
#include <iostream>


TEST_CASE("naive poly add") {
    using mont::u32;
    using mont::u64;
    using Field = bn256_fr::Element;
    using Number = mont::Number<Field::LIMBS>;

    std::cout << "testing the naive poly add" << std::endl;
    auto len = 1 << 24;
    auto size = len * Field::LIMBS;
    Field * a, *b, *dst;
    a = (Field *) new u32 [size];
    b = (Field *) new u32 [size];
    dst = (Field *) new u32 [size];

    for (u64 i = 0; i < len; i++) {
        a[i] = Field::host_random();
        b[i] = Field::host_random();
    }

    u32 *a_d, *b_d, *dst_d;
    cudaMalloc(&a_d, len * Field::LIMBS * sizeof(u32));
    cudaMalloc(&b_d, len * Field::LIMBS * sizeof(u32));
    cudaMalloc(&dst_d, len * Field::LIMBS * sizeof(u32));
    cudaMemcpy(a_d, a, len * Field::LIMBS * sizeof(u32), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, len * Field::LIMBS * sizeof(u32), cudaMemcpyHostToDevice);

    poly::NaiveAdd<Field><<<((len - 1) / 1024) + 1, 1024>>>(a_d, b_d, dst_d, len);

    u32 block = 1024;
    u32 grid = (len - 1) / block + 1;
    
    const int io_group = 8;
    u32 shared_size = (sizeof(cub::WarpExchange<u32, io_group, io_group>::TempStorage) * (block / io_group));

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    poly::Add<Field, io_group><<<grid, block, shared_size >>>(a_d, b_d, dst_d, len);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "Time: " << milliseconds << "ms" << std::endl;

    cudaMemcpy(dst, dst_d, len * Field::LIMBS * sizeof(u32), cudaMemcpyDeviceToHost);

    for (u64 i = 0; i < len; i++) {
        CHECK(dst[i] == a[i] + b[i]);
    }
}

TEST_CASE("naive poly mul") {
    using mont::u32;
    using mont::u64;
    using Field = bn256_fr::Element;
    using Number = mont::Number<Field::LIMBS>;

    std::cout << "testing the naive poly mul" << std::endl;
    auto len = 1 << 24;
    auto size = len * Field::LIMBS;
    Field * a, *b, *dst;
    a = (Field *) new u32 [size];
    b = (Field *) new u32 [size];
    dst = (Field *) new u32 [size];

    for (u64 i = 0; i < len; i++) {
        a[i] = Field::host_random();
        b[i] = Field::host_random();
    }

    u32 *a_d, *b_d, *dst_d;
    cudaMalloc(&a_d, len * Field::LIMBS * sizeof(u32));
    cudaMalloc(&b_d, len * Field::LIMBS * sizeof(u32));
    cudaMalloc(&dst_d, len * Field::LIMBS * sizeof(u32));
    cudaMemcpy(a_d, a, len * Field::LIMBS * sizeof(u32), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, len * Field::LIMBS * sizeof(u32), cudaMemcpyHostToDevice);

    poly::NaiveMul<Field><<<((len - 1) / 1024) + 1, 1024>>>(a_d, b_d, dst_d, len);

    u32 block = 1024;
    u32 grid = (len - 1) / block + 1;
    
    const int io_group = 8;
    u32 shared_size = (sizeof(cub::WarpExchange<u32, io_group, io_group>::TempStorage) * (block / io_group));

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    poly::Mul<Field, io_group><<<grid, block, shared_size >>>(a_d, b_d, dst_d, len);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "Time: " << milliseconds << "ms" << std::endl;

    cudaMemcpy(dst, dst_d, len * Field::LIMBS * sizeof(u32), cudaMemcpyDeviceToHost);

    for (u64 i = 0; i < len; i++) {
        CHECK(dst[i] == a[i] * b[i]);
    }
}