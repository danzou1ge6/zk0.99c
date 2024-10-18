#pragma once
#include "../include/nlohmann/json.hpp"
#include "field_type.h"
#include "memory.h"
#include <map>
#include <vector>
#include <variant>
#include <set>
#include <future>
#include <cuda_runtime.h>

namespace runtime {
    
    struct RunInfo {
        int mem_size;
        float compute_intensity;
        RunInfo() = default;
        RunInfo(const nlohmann::json& j);
        friend auto operator < (const RunInfo& a, const RunInfo& b) -> bool;
    };

    struct Node {
        enum Status {
            WAITING,
            RUNNING,
            FINISHED
        };

        struct KernelInfo {
            std::string kernel_name;
            int input_num;
            std::vector<int> inputs;
            KernelInfo() = default;
            KernelInfo(const nlohmann::json& j);
        };

        struct MemInfo {
            MemType type;
            size_t size;
            unsigned int register_id;
            MemInfo() = default;
            MemInfo(const nlohmann::json& j);
        };

        struct CopyInfo {
            cudaMemcpyKind type;
            size_t size;
            CopyInfo() = default;
            CopyInfo(const nlohmann::json& j);
        };

        struct NttInfo {
            enum NttType {
                NTT,
                INVERSE,
                COEFF_TO_EXT,
                EXT_TO_COEFF
            };
            NttType type;
            FieldType field;
            unsigned int logn;
            NttInfo() = default;
            NttInfo(const nlohmann::json& j);
        };

        std::variant<KernelInfo, MemInfo, CopyInfo, NttInfo> info; // the information of the node
        std::vector<int> last, next; // the last and next nodes
        std::set<int> wait_for; // the nodes that this node is waiting for
        const bool is_cuda; // whether this node is a cuda node
        std::variant<std::future<void>, cudaStream_t> stream; // the stream of the node
        Status status;
        RunInfo run_info;

        Node(const nlohmann::json& j);
        void sync();
    
    };

    class ComputeGraph {
        private:
            std::vector<Node> nodes;
            std::multimap<RunInfo,int> available_nodes;
        public:
            ComputeGraph() = default;
            ComputeGraph(const nlohmann::json& j);
            auto get_next() -> int const;
            auto get_next(int memory, float compute_intensity) -> int const;
            auto run_node(int id) -> void;
            auto finish() -> bool const;
    };
}