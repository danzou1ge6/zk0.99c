#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "../src/poly_eval.cuh"
#include "../../mont/src/bn254_fr.cuh"

typedef bn254_fr::Element Field;

Field* gen_poly(uint len) {
    Field* poly = new Field[len];
    for (uint i = 0; i < len; i++) {
        poly[i] = Field::host_random();
    }
    return poly;
}

Field eval_cpu(Field * poly, Field x, uint len) {
    Field res = Field::zero();
    for (uint i = 0; i < len; i++) {
        res = res + poly[i] * x.pow(i);
    }
    return res;
}

TEST_CASE("Naive eval") {
    auto len = 1 << 24;
    auto poly = gen_poly(len);
    auto x = Field::host_random();
    uint* poly_d;
    cudaMalloc(&poly_d, len * Field::LIMBS * sizeof(uint));
    cudaMemcpy(poly_d, poly, len * Field::LIMBS * sizeof(uint), cudaMemcpyHostToDevice);
    uint *res_d;
    cudaMalloc(&res_d, Field::LIMBS * sizeof(uint));
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    poly::NaiveEval(poly_d, poly_d, res_d, x, len, 0);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "Time: " << milliseconds << "ms" << std::endl;
    Field res;
    cudaMemcpy(&res, res_d, Field::LIMBS * sizeof(uint), cudaMemcpyDeviceToHost);
    auto res_cpu = eval_cpu(poly, x, len);
    CHECK(res == res_cpu);
    cudaFree(poly_d);
    cudaFree(res_d);
    delete [] poly;
}

TEST_CASE("eval") {
    auto len = 1 << 24;
    auto poly = gen_poly(len);
    auto x = Field::host_random();
    uint* poly_d;
    cudaMalloc(&poly_d, len * Field::LIMBS * sizeof(uint));
    cudaMemcpy(poly_d, poly, len * Field::LIMBS * sizeof(uint), cudaMemcpyHostToDevice);
    uint *res_d;
    cudaMalloc(&res_d, Field::LIMBS * sizeof(uint));
    uint *temp_buf;
    cudaMalloc(&temp_buf, len * Field::LIMBS * sizeof(uint));
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    poly::Eval(poly_d, temp_buf, res_d, x, len, 0);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "Time: " << milliseconds << "ms" << std::endl;
    Field res;
    cudaMemcpy(&res, res_d, Field::LIMBS * sizeof(uint), cudaMemcpyDeviceToHost);
    auto res_cpu = eval_cpu(poly, x, len);
    CHECK(res == res_cpu);

    cudaFree(poly_d);
    cudaFree(res_d);
    delete [] poly;
}