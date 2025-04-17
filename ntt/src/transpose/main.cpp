#include "matrix_transpose.h"
#include <iostream>
#include <chrono>
#include <iomanip>

template<size_t N>
void runTest(size_t rows, size_t cols) {
    std::cout << "\nRunning test with " << N << " byte elements" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Matrix size: " << rows << "x" << cols << std::endl;
    const size_t total_size = rows * cols * sizeof(LargeInteger<N>);
    std::cout << "Total memory required: " << (total_size / (1024.0 * 1024.0)) << " MB" << std::endl;
    
    // 分配内存
    auto* srcMat = new LargeInteger<N>[rows * cols];
    auto* dstMat = new LargeInteger<N>[rows * cols];
    
    if (!srcMat || !dstMat) {
        std::cerr << "Memory allocation failed!" << std::endl;
        delete[] srcMat;
        delete[] dstMat;
        return;
    }
    
    std::cout << "Memory allocated successfully" << std::endl;
    
    // 初始化源矩阵
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            auto& elem = srcMat[i * cols + j];
            for (size_t k = 0; k < N; k++) {
                elem.data[k] = (i * cols + j + k) % 256;
            }
        }
    }
    
    std::cout << "Matrix initialized, starting transpose..." << std::endl;
    
    // 计时并执行转置
    auto start = std::chrono::high_resolution_clock::now();
    transpose<N>(srcMat, dstMat, rows, cols);
    auto end = std::chrono::high_resolution_clock::now();
    
    // 输出运行时间
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Transpose completed in " << elapsed.count() << " seconds" << std::endl;
    
    // 验证结果正确性
    bool correct = true;
    for (size_t i = 0; i < rows && correct; i++) {
        for (size_t j = 0; j < cols && correct; j++) {
            const auto& srcElem = srcMat[i * cols + j];
            const auto& dstElem = dstMat[j * rows + i];
            
            for (size_t k = 0; k < N; k++) {
                if (srcElem.data[k] != dstElem.data[k]) {
                    correct = false;
                    std::cerr << "Mismatch at (" << i << "," << j << ") byte " << k 
                             << ": src=" << (int)srcElem.data[k] 
                             << " dst=" << (int)dstElem.data[k] << std::endl;
                    break;
                }
            }
        }
    }
    
    std::cout << "Transpose " << (correct ? "correct" : "incorrect") << std::endl;
    
    // 计算带宽
    double dataSize = rows * cols * N * 2.0; // 读+写
    double bandwidthGB = (dataSize / (1024 * 1024 * 1024)) / elapsed.count(); // GB/s
    
    std::cout << "Memory bandwidth: " << bandwidthGB << " GB/s" << std::endl;
    
    // 释放内存
    delete[] srcMat;
    delete[] dstMat;
}

int main() {
    constexpr size_t ROWS = 32;
    constexpr size_t COLS = 1 << 20;  // 2^20
    
    // 测试不同大小的元素
    runTest<32>(ROWS, COLS);
    runTest<64>(ROWS, COLS);
    runTest<96>(ROWS, COLS);
    
    return 0;
}