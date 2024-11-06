#include "memory.cuh"
#include <stdexcept>
#include <string>

namespace runtime {
    auto get_mem_type(const std::string &mem_type) -> MemType {
        if (mem_type == "host") {
            return MemType::HOST;
        } else if (mem_type == "device") {
            return MemType::DEVICE;
        } else if (mem_type == "managed") {
            return MemType::MANAGED;
        } else {
            throw std::invalid_argument("Invalid memory type: " + mem_type);
        }
    }

    auto operator<<(std::ostream &os, const MemType &mem_type) -> std::ostream & {
        switch (mem_type) {
            case MemType::HOST:
                os << std::string("host");
                break;
            case MemType::DEVICE:
                os << std::string("device");
                break;
            case MemType::MANAGED:
                os << std::string("managed");
                break;
        }
        return os;
    }

    auto operator<<(std::ostream &os, const CopyType &copy_type) -> std::ostream & {
        switch (copy_type) {
            case CopyType::H2H:
                os << std::string("H2H");
                break;
            case CopyType::H2D:
                os << std::string("H2D");
                break;
            case CopyType::D2H:
                os << std::string("D2H");
                break;
            case CopyType::D2D:
                os << std::string("D2D");
                break;
        }
        return os;
    }
}