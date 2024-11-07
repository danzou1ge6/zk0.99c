#include "memory.cuh"
#include <mutex>
#include <stdexcept>
#include <string>
#include "error.cuh"
#include <cassert>

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

    MemoryPool::MemoryPool(
        usize max_buffer_size, 
        usize max_pool_size,
        int device_id,
        MemType mem_type,
        ExpandType expand_type
    ) : max_buffer_size(max_buffer_size), max_pool_size(max_pool_size)
    , mem_type(mem_type), expand_type(expand_type), allocted_size(0), device_id(device_id) {
        assert(max_buffer_size <= max_pool_size);
        if (mem_type == MemType::DEVICE) {
            assert(device_id >= 0);
        } else {
            assert(device_id == -1);
        }
        mtx.lock();
        auto raw = alloc_buffer();
        Chunk chunk = {raw.first, raw.second, false, mem_type, 0, this};
        std::list<Chunk> buffer = {chunk};
        buffers.push_back(buffer);
    }

    auto MemoryPool::alloc_buffer() -> std::pair<byte *, usize> {
        byte *buffer;
        usize buffer_size = std::min(max_buffer_size, max_pool_size - allocted_size);
        if (mem_type == MemType::HOST) {
            CUDA_CHECK_EXIT(cudaHostAlloc(&buffer, buffer_size, cudaHostAllocDefault));
        } else if (mem_type == MemType::DEVICE) {
            CUDA_CHECK_EXIT(cudaSetDevice(device_id));
            CUDA_CHECK_EXIT(cudaMalloc(&buffer, buffer_size));
        } else if (mem_type == MemType::MANAGED) {
            CUDA_CHECK_EXIT(cudaMallocManaged(&buffer, buffer_size));
        }
        allocted_size += buffer_size;
        return std::make_pair(buffer, buffer_size);
    }

    auto MemoryPool::extend_pool() -> bool {
        if (allocted_size >= max_pool_size) {
            return false;
        }
        mtx.lock();
        if (expand_type == ExpandType::INCREASE) {
            auto raw = alloc_buffer();
            Chunk chunk = {raw.first, raw.second, false, mem_type, buffers.size(), this};
            std::list<Chunk> buffer = {chunk};
            buffers.push_back(alloc_buffer());
        } else {
            usize size = buffers.size();
            for (usize i = 0; i < size; i++) {
                if (allocated_size >= max_pool_size) {
                    break;
                }
                auto raw = alloc_buffer();
                Chunk chunk = {raw.first, raw.second, false, mem_type, buffers.size(), this};
                std::list<Chunk> buffer = {chunk};
                buffers.push_back(buffer);
            }
        }
        mtx.unlock();
        return true;
    }

    auto MemoryPool::get_chunk(usize size) -> std::optional<Chunk> {
        assert(size <= max_buffer_size);

        mtx.lock();
        int buffer_index = -1;
        int chunk_index = -1;
        usize min_size = max_buffer_size;

        for (usize i = 0; i < buffers.size(); i++) {
            int j = 0;
            for (auto it = buffers[i].begin(); it != buffers[i].end(); it++) {
                if (it->size >= size && it->size < min_size) {
                    buffer_index = i;
                    chunk_index = j;
                    min_size = it->size;
                }
                j++;
            }
        }

        if (buffer_index == -1) {
            mtx.unlock();
            if (extend_pool()) return get_chunk(size);
            return std::nullopt;
        }

        Chunk res;

        int j = 0;
        for (auto it = buffers[buffer_index].begin(); it != buffers[buffer_index].end(); it++) {
            if (j == chunk_index) {
                res = *it;
                res.size = size;
                if (it->size > size) {
                    it->p += size;
                    it->size -= size;
                } else {
                    buffers[buffer_index].erase(it);
                }
                break;
            }
            j++;
        }
        mtx.unlock();
        return res;
    }

    MemoryPool::~MemoryPool() {
        mtx.lock();
        for (usize i = 0; i < buffers.size(); i++) {
            for (auto it = buffers[i].begin(); it != buffers[i].end(); it++) {
                if (mem_type == MemType::HOST) {
                    CUDA_CHECK_EXIT(cudaFreeHost(it->p));
                } else {
                    if (device_id != -1) CUDA_CHECK_EXIT(cudaSetDevice(device_id));
                    CUDA_CHECK_EXIT(cudaFree(it->p));
                }
            }
        }
        mtx.unlock();
    }

    auto MemoryPool::return_chunk(Chunk &chunk) -> void {
        assert(chunk.pool == this);

        mtx.lock();
        chunk.dirty = false;
        int buffer_index = chunk.buffer_id;
        for (auto it = buffers[buffer_index].begin(); ; it++) {
            if (it == buffers[buffer_index].end() && it == buffers[buffer_index].begin()) {
                buffers[buffer_index].push_back(chunk);
                break;
            }
            auto next = std::next(it);
        
            if (
                (it->p + it->size <= chunk.p) &&
                (next == buffers[buffer_index].end() || chunk.p + chunk.size <= next->p)
            ) {
                auto target = buffers[buffer_index].insert(next, chunk);                
                auto last = std::prev(target);
                // merge with the previous chunk
                while(target != buffers[buffer_index].begin()) {
                    if (last->p + last->size == target->p) {
                        target->p = last->p;
                        target->size += last->size;
                        buffers[buffer_index].erase(last);
                        last = std::prev(target);
                    } else {
                        break;
                    }
                }
                // merge with the next chunk
                auto next = std::next(target);
                while(next != buffers[buffer_index].end()) {
                    if (target->p + target->size == next->p) {
                        target->size += next->size;
                        buffers[buffer_index].erase(next);
                        next = std::next(target);
                    } else {
                        break;
                    }
                }
                break;
            }
            
            if (it == buffers[buffer_index].end()) {
                throw std::runtime_error("Invalid chunk");
            }
        }
        mtx.unlock();
    }
}