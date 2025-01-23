#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "../src/kate_division.cuh"
#include "../../mont/src/bn254_fr.cuh"
#include <iostream>
#include <chrono>

using mont::u32;
using mont::u64;
using Field = bn254_fr::Element;
using Number = mont::Number<Field::LIMBS>;

TEST_CASE("recrusive kate division") {
    std::cout << "testing the recrusive kate division" << std::endl;
    Field * p, *q, *q_truth;
    u32 len = 1233237;
    p = new Field [len];
    q = new Field [len];
    q_truth = new Field [len];

    for (u64 i = 0; i < len; i++) {
        p[i] = Field::host_random();
    }

    Field b = Field::host_random();

    // start timer
    auto start = std::chrono::high_resolution_clock::now();
    poly::kate_divison(len, p, b, q_truth);
    // end timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "baseline Time: " << elapsed.count() << "s" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    poly::recrusive_kate_divison<Field>(len, p, b, q);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Recrusive Time: " << elapsed.count() << "s" << std::endl;

    for (u64 i = 0; i <= len - 1; i++) {
        CHECK(q[i] == q_truth[i]);
    }

    delete[] p;
    delete[] q;
    delete[] q_truth;
}

TEST_CASE("gpu kate division") {
    std::cout << "testing the gpu kate division" << std::endl;
    Field * p, *q, *q_truth;
    u32 log_len = 24;
    u32 len = 1 << log_len;
    p = new Field [len];
    q = new Field [len];
    q_truth = new Field [len];

    for (u64 i = 0; i < len; i++) {
        p[i] = Field::host_random();
    }

    Field b = Field::host_random();

    // start timer
    auto start = std::chrono::high_resolution_clock::now();
    poly::kate_divison(len, p, b, q_truth);
    // end timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "baseline Time: " << elapsed.count() << "s" << std::endl;

    Field *p_d, *q_d;
    cudaMalloc(&p_d, len * sizeof(Field));
    cudaMalloc(&q_d, len * sizeof(Field));
    cudaMemcpy(p_d, p, len * sizeof(Field), cudaMemcpyHostToDevice);

    poly::gpu_kate_division<Field>(log_len, p_d, b, q_d);

    cudaMemcpy(q, q_d, len * sizeof(Field), cudaMemcpyDeviceToHost);

    for (u64 i = 0; i <= len - 1; i++) {
        CHECK(q[i] == q_truth[i]);
    }

    delete[] p;
    delete[] q;
    delete[] q_truth;
}