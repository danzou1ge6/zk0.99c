#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <iostream>
#include "../src/bellperson_ntt.cuh"
#include "../src/naive_ntt.cuh"
#include "../src/self_sort_in_place_ntt.cuh"
#include "../../mont/src/bls12381_fr.cuh"

#include <iostream>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <sstream>
#include <vector>

// # BLS12-381
// #[PrimeFieldModulus = "52435875175126190479447740508185965837690552500527637822603658699938581184513"]
// #[PrimeFieldGenerator = "7"]

const uint unit[] = BIG_INTEGER_CHUNKS8(0, 0, 0, 0, 0, 0, 0, 7);

const long long WORDS = 8;

const int max_deg = 20;

void exec(const char* cmd, uint * input, uint * output, uint len) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }

    // 解析结果字符串为整数列表
    std::vector<uint> numbers;
    std::stringstream ss(result);
    std::string token;

    for (long long i = 0; i < WORDS * len; i++) {
        std::getline(ss, token, ' ');
        try {
            input[i] = std::stoul(token);  // 将字符串转换为整数并添加到vector中
        } catch (std::invalid_argument& e) {
            // 处理无效输入
            std::cerr << "Invalid number in input: " << token << std::endl;
        }
    }

    for (long long i = 0; i < WORDS * len; i++) {
        std::getline(ss, token, ' ');
        try {
            output[i] = std::stoul(token);  // 将字符串转换为整数并添加到vector中
        } catch (std::invalid_argument& e) {
            // 处理无效输入
            std::cerr << "Invalid number in input: " << token << std::endl;
        }
    }

}

TEST_CASE("testing the naive GPU NTT implementation") {

    std::cout << "testing the naive GPU NTT implementation" << std::endl;

    for (int logn = 1; logn <= max_deg; logn++) {
        int n = 1 << logn;

        std::cout << "testing n = 2^" << logn << ":" << std::endl;

        std::stringstream script;

        script << "python3 " << PROJECT_ROOT << "/ntt/tests/generate_ntt_sample.py --log_len " << logn;

        uint *input, *answer;

        input = new uint[WORDS * n];
        answer = new uint[WORDS * n];

        exec(script.str().c_str(), input, answer, n);

        ntt::naive_ntt<bls12381_fr::Element> naive(unit, logn, true);
        naive.ntt(input);
        std::cout << "naive: " << naive.milliseconds << "ms" << std::endl;

        auto err = cudaGetLastError();
        CHECK(err == cudaSuccess);
        if (err != cudaSuccess) {
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        for (long long i = 0; i < n; i++) {
            for (int j = 0; j < WORDS; j++) {
                CHECK(input[i * WORDS + j] == answer[i * WORDS + j]);
            }
        }

        delete[] input;
        delete[] answer;
    }
}

TEST_CASE("testing the Bellperson NTT implementation") {

    std::cout << "testing the Bellperson NTT implementation" << std::endl;

    for (int logn = 1; logn <= max_deg; logn++) {
        int n = 1 << logn;

        std::cout << "testing n = 2^" << logn << ":" << std::endl;

        std::stringstream script;

        script << "python3 " << PROJECT_ROOT << "/ntt/tests/generate_ntt_sample.py --log_len " << logn;

        uint *input, *answer;

        input = new uint[WORDS * n];
        answer = new uint[WORDS * n];

        exec(script.str().c_str(), input, answer, n);

        ntt::bellperson_ntt<bls12381_fr::Element> bellperson(unit, logn, true);
        bellperson.ntt(input);
        std::cout << "bellperson: " << bellperson.milliseconds << "ms" << std::endl;

        auto err = cudaGetLastError();
        CHECK(err == cudaSuccess);
        if (err != cudaSuccess) {
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        for (long long i = 0; i < n; i++) {
            for (int j = 0; j < WORDS; j++) {
                CHECK(input[i * WORDS + j] == answer[i * WORDS + j]);
            }
        }

        delete[] input;
        delete[] answer;
    }
}

TEST_CASE("testing the SSIP NTT implementation") {

    std::cout << "testing the SSIP NTT implementation" << std::endl;

    for (int logn = 1; logn <= max_deg; logn++) {
        int n = 1 << logn;

        std::cout << "testing n = 2^" << logn << ":" << std::endl;

        std::stringstream script;

        script << "python3 " << PROJECT_ROOT << "/ntt/tests/generate_ntt_sample.py --log_len " << logn;

        uint *input, *answer;

        input = new uint[WORDS * n];
        answer = new uint[WORDS * n];

        exec(script.str().c_str(), input, answer, n);

        ntt::self_sort_in_place_ntt<bls12381_fr::Element> SSIP(unit, logn, true);
        SSIP.ntt(input);
        std::cout << "SSIP: " << SSIP.milliseconds << "ms" << std::endl;

        auto err = cudaGetLastError();
        CHECK(err == cudaSuccess);
        if (err != cudaSuccess) {
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        for (long long i = 0; i < n; i++) {
            for (int j = 0; j < WORDS; j++) {
                CHECK(input[i * WORDS + j] == answer[i * WORDS + j]);
            }
        }

        delete[] input;
        delete[] answer;
    }
}