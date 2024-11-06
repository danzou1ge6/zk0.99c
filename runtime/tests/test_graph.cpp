#include <iostream>
#include <fstream>
#include <string>
#include "../src/graph.cuh"
#include <nlohmann/json.hpp>

int main() {
    nlohmann::json j;
    std::string path = PROJECT_ROOT;
    path += "/runtime/tests/test_graph.json";
    std::ifstream i(path);
    if (!i.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }
    i >> j;
    runtime::ComputeGraph cg(j["nodes"]);
    std::cout << cg << std::endl;
    return 0;
}