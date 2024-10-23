#pragma once
#include <vector>
#include <string>
#include "basic_types.h"

namespace runtime {
    enum MemType {
        HOST,
        DEVICE,
        MANAGED,
        PINNED
    };

    auto get_mem_type(const std::string &mem_type) -> MemType;
    
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
}
