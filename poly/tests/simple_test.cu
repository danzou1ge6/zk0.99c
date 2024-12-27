#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "../src/poly.cuh"
#include "../../mont/src/bn254_fr.cuh"
#include <iostream>

using mont::u32;
using mont::u64;
using Field = bn254_fr::Element;
using Number = mont::Number<Field::LIMBS>;

u64 len = 1 << 24;

TEST_CASE("naive poly add") {
    std::cout << "testing the naive poly add" << std::endl;
    Field * a, *dst;
    a = new Field [len];
    dst = new Field [len];

    for (u64 i = 0; i < len; i++) {
        a[i] = Field::host_random();
    }

    u32 *a_d;
    cudaMalloc(&a_d, len * Field::LIMBS * sizeof(u32));

    u32 block = 1024;
    u32 grid = (len - 1) / block + 1;
    poly::NaiveAdd<Field><<<grid, block >>>(a_d, a_d, a_d, len);

    cudaMemcpy(a_d, a, len * Field::LIMBS * sizeof(u32), cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    poly::NaiveAdd<Field><<<grid, block >>>(a_d, a_d, a_d, len);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "Time: " << milliseconds << "ms" << std::endl;

    cudaMemcpy(dst, a_d, len * Field::LIMBS * sizeof(u32), cudaMemcpyDeviceToHost);

    for (u64 i = 0; i < len; i++) {
        CHECK(dst[i] == a[i] + a[i]);
    }

    delete[] a;
    delete[] dst;
    cudaFree(a_d);
}

TEST_CASE("poly add") {
    std::cout << "testing the poly add" << std::endl;
    Field * a, *dst;
    a = new Field [len];
    dst = new Field [len];

    for (u64 i = 0; i < len; i++) {
        a[i] = Field::host_random();
    }

    u32 *a_d;
    cudaMalloc(&a_d, len * Field::LIMBS * sizeof(u32));
    cudaMemcpy(a_d, a, len * Field::LIMBS * sizeof(u32), cudaMemcpyHostToDevice);

    u32 block = 1024;
    u32 grid = (len - 1) / block + 1;
    const int io_group = 8;
    u32 shared_size = (sizeof(cub::WarpExchange<u32, io_group, io_group>::TempStorage) * (block / io_group));

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    poly::Add<Field, io_group><<<grid, block, shared_size >>>(a_d, a_d, a_d, len);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "Time: " << milliseconds << "ms" << std::endl;

    cudaMemcpy(dst, a_d, len * Field::LIMBS * sizeof(u32), cudaMemcpyDeviceToHost);

    for (u64 i = 0; i < len; i++) {
        CHECK(dst[i] == a[i] + a[i]);
    }

    delete[] a;
    delete[] dst;
    cudaFree(a_d);
}

TEST_CASE("naive poly mul") {
    std::cout << "testing the naive poly mul" << std::endl;
    Field * a, *dst;
    a = new Field [len];
    dst = new Field [len];

    for (u64 i = 0; i < len; i++) {
        a[i] = Field::host_random();
    }

    u32 *a_d;
    cudaMalloc(&a_d, len * Field::LIMBS * sizeof(u32));

    u32 block = 1024;
    u32 grid = (len - 1) / block + 1;

    cudaMemcpy(a_d, a, len * Field::LIMBS * sizeof(u32), cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    poly::NaiveMul<Field><<<grid, block >>>(a_d, a_d, a_d, len);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "Time: " << milliseconds << "ms" << std::endl;

    cudaMemcpy(dst, a_d, len * Field::LIMBS * sizeof(u32), cudaMemcpyDeviceToHost);

    for (u64 i = 0; i < len; i++) {
        CHECK(dst[i] == a[i] * a[i]);
    }

    delete[] a;
    delete[] dst;
    cudaFree(a_d);
}

TEST_CASE("poly mul") {
    std::cout << "testing the poly mul" << std::endl;
    Field * a, *dst;
    a = new Field [len];
    dst = new Field [len];

    for (u64 i = 0; i < len; i++) {
        a[i] = Field::host_random();
    }

    u32 *a_d;
    cudaMalloc(&a_d, len * Field::LIMBS * sizeof(u32));
    cudaMemcpy(a_d, a, len * Field::LIMBS * sizeof(u32), cudaMemcpyHostToDevice);

    u32 block = 1024;
    u32 grid = (len - 1) / block + 1;
    const int io_group = 8;
    u32 shared_size = (sizeof(cub::WarpExchange<u32, io_group, io_group>::TempStorage) * (block / io_group));

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    poly::Mul<Field, io_group><<<grid, block, shared_size >>>(a_d, a_d, a_d, len);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "Time: " << milliseconds << "ms" << std::endl;

    cudaMemcpy(dst, a_d, len * Field::LIMBS * sizeof(u32), cudaMemcpyDeviceToHost);

    for (u64 i = 0; i < len; i++) {
        CHECK(dst[i] == a[i] * a[i]);
    }

    delete[] a;
    delete[] dst;
    cudaFree(a_d);
}