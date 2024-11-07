#pragma once
#include <mutex>
#include <optional>
#include <vector>
#include <list>
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

    class MemoryPool;

    struct Chunk {
        byte *p = nullptr; // pointer to the chunk
        usize size; // size of the chunk
        bool dirty; // is it dirty?
        MemType host_type; // memory type
        usize buffer_id; // which buffer does it belong to in the memory pool?
        MemoryPool *pool; // which memory pool does it belong to?
    };
    
    struct Buffer {
        std::optional<Chunk> chunk, chunk_d; // chunk for cpu, chunk_d for gpu
        MemType host_type; // memory type
    };

    // used to manage all the buffers
    struct Instance {
        std::vector<Buffer> buffers;
    };

    // used to manage the memory pool
    // buffer_size: the size of each buffer, should be set to the maximum size of all used buffers
    // we use a lock to gard the memory pool
    class MemoryPool {
    public:
        enum ExpandType {
            DOUBLE,
            INCREASE
        };
        auto get_chunk(usize size) -> std::optional<Chunk>; // we currently use the best fit algorithm
        auto return_chunk(Chunk &chunk) -> void;
        MemoryPool(usize buffer_size, usize max_pool_size, int device_id, MemType mem_type = MemType::HOST, ExpandType expand_type = ExpandType::INCREASE);
        ~MemoryPool();
    private:
        std::vector<std::list<Chunk>> buffers;
        std::mutex mtx;
        usize allocted_size;
        const usize max_buffer_size;
        const usize max_pool_size;
        const MemType mem_type;
        ExpandType expand_type;
        int device_id; // -1 for host, >= 0 for device

        auto alloc_buffer() -> std::pair< byte *, usize >;
        auto extend_pool() -> bool;
    };
}
