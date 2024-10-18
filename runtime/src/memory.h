#pragma once
#include <vector>

namespace runtime {
    enum MemType {
        HOST,
        DEVICE,
        MANAGED,
        PINNED
    };
    
    struct Buffer {
        void *p, *p_d; // pointer to the buffer, p for cpu, p_d for gpu
        std::size_t size;
        int device_id; // which device?
        bool on_cpu; // is it on cpu?
        bool dirty; // is it dirty?
        MemType host_type; // memory type
    };

    // used to manage all the buffers
    struct Instance {
        std::vector<Buffer> buffers;
    };
}
