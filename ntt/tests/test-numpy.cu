#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include "small_field.cuh"
namespace py = pybind11;

int main() {
    py::scoped_interpreter guard{};

    int lgn = 5;
    int lgp = 2;
    int lgq = lgn - lgp;

    small_field::Element *a, *b;

    a = new small_field::Element[1 << lgn];
    b = new small_field::Element[1 << lgn];

    
    for (int i = 0; i < (1 << lgp); i++) {
        for (int j = 0; j < (1 << lgq); j++) {
            for (int k = 0; k < 8; k++) {
                a[i * (1 << lgq) + j].n.limbs[k] = i * 8 * (1 << lgq) + j * 8 + k;
            }
        }
    }

    for (int i = 0; i < (1 << lgp); i++) {
        for (int j = 0; j < (1 << lgq); j++) {
            for (int k = 0; k < 8; k++) {
                printf("%d ", a[i * (1 << lgq) + j].n.limbs[k]);
            }
            printf("\t");
        }
        printf("\n");
    }

    auto src = py::memoryview::from_buffer((uint*)a, {(1 << lgp), 1<<lgq, 8}, {sizeof(int) * 8 * (1 << lgq), sizeof(int) *8, sizeof(int)});
    auto dst = py::memoryview::from_buffer((uint*)b, {1<<lgq,(1 << lgp) , 8}, {sizeof(int) * 8 * (1 << lgp), sizeof(int) *8, sizeof(int)});


    py::exec(R"(
        import numpy as np
        def transpose(src, dst, a, b, c):
            x = np.frombuffer(src, dtype=np.uint32)
            x = x.reshape(a,b,c)
            y = np.frombuffer(dst, dtype=np.uint32)
            y = y.reshape(b,a,c)
            np.copyto(y, np.transpose(x, (1, 0, 2)))
            
    )");


    // 获取 Python 中的 transpose 函数
    py::object transpose_func = py::module::import("__main__").attr("transpose");

    transpose_func(src, dst, 1<<lgp, 1<<lgq, 8);
    for (int i = 0; i < (1 << lgq); i++) {
        for (int j = 0; j < (1 << lgp); j++) {
            for (int k = 0; k < 8; k++) {
                printf("%d ", b[i * (1 << lgp) + j].n.limbs[k]);
            }
            printf("\t");
        }
        printf("\n");
    }

    return 0;
}