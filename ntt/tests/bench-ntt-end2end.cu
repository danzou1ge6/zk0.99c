#include "../src/self_sort_in_place_ntt.cuh"
#include "../../mont/src/bn254_fr.cuh"

using namespace ntt;
typedef bn254_fr::Element Field;

int main (int argc, char *argv[]) {
    if (argc!= 2) {
      std::cout << "Usage: " << argv[0] << " <k>" << std::endl;
      return 1;
    }

    int k = atoi(argv[1]);

    auto omega = Field::host_random();
    auto config = self_sort_in_place_ntt<Field>::SSIP_config();
    self_sort_in_place_ntt<Field> ntt(reinterpret_cast<u32*>(&omega), k, true, 1, false, false, nullptr, nullptr, config);
    ntt.to_gpu();
    
    Field *data;
    cudaHostAlloc(&data, (1ll << k) * sizeof(Field), cudaHostAllocDefault);
    for (int i = 0; i < (1ll << k); i++) {
        data[i] = Field::host_random();
    }

    // warm up, because the jit compilation is slow
    for (int i = 0; i < 10; i ++)
      ntt.ntt(reinterpret_cast<u32*>(data), 0, 0, false);


    int n = 10;
    float e2e_acc = 0;
    float comp_acc = 0;
    for (int i = 0; i < n; i++) {
        ntt.ntt(reinterpret_cast<u32*>(data), 0, 0, false);
        e2e_acc += ntt.total_milliseconds;
        comp_acc += ntt.milliseconds;
    }

    float e2e = e2e_acc / n;
    float comp = comp_acc / n;
    
    std::cout << "End-to-end time  : " << e2e << " ms" << std::endl;
    std::cout << "Computation time : " << comp << " ms" << std::endl;
    
    cudaFreeHost(data);
    return 0;
}

