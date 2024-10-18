#include "../include/nlohmann/json.hpp"
#include <iostream>

int main() {
    nlohmann::json j;
    std::cin >> j;
    std::cout << j["inputs"] << std::endl;
    std::cout << j["nodes"] << std::endl;
    return 0;
}