#include "memory.h"
#include <stdexcept>

namespace runtime {
    auto get_mem_type(const std::string &mem_type) -> MemType {
        if (mem_type == "host") {
            return MemType::HOST;
        } else if (mem_type == "device") {
            return MemType::DEVICE;
        } else if (mem_type == "managed") {
            return MemType::MANAGED;
        } else if (mem_type == "pinned") {
            return MemType::PINNED;
        } else {
            throw std::invalid_argument("Invalid memory type: " + mem_type);
        }
    }
}