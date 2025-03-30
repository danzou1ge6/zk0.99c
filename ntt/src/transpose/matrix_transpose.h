#pragma once
#include "large_integer.h"

template<size_t N>
void transpose(const LargeInteger<N>* src, LargeInteger<N>* dst, size_t rows, size_t cols, 
               size_t maxThreads = 0);  // maxThreads=0 表示自动选择