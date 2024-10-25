#pragma once
#include <atomic>
#include <vector>
#include <string>
#include "basic_types.h"

namespace runtime {
    enum MemType {
        HOST,
        DEVICE,
        MANAGED
    };
    auto operator<<(std::ostream &os, const MemType &mem_type) -> std::ostream &;
    auto get_mem_type(const std::string &mem_type) -> MemType;

    enum CopyType {
        H2H,
        H2D,
        D2H,
        D2D
    };
    auto operator<<(std::ostream &os, const CopyType &copy_type) -> std::ostream &;
    
    struct Buffer {
        void *p, *p_d; // pointer to the buffer, p for cpu, p_d for gpu
        usize size;
        int device_id; // which device?
        bool on_cpu; // is it on cpu?
        int dirty; // is it dirty?
        MemType host_type; // memory type
    };

    // used to manage all the buffers
    struct Instance {
        std::vector<Buffer> buffers;
    };

    class MemoryPool {
        public:
    };
}
