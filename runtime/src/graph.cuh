#pragma once
#include <nlohmann/json.hpp>
#include "field_type.h"
#include "memory.cuh"
#include <map>
#include <vector>
#include <variant>
#include <set>
#include <future>
#include <cuda_runtime.h>
#include "basic_types.h"
#include <iostream>

namespace runtime {

    struct RunInfo {
        usize mem_size;
        float compute_intensity;
        RunInfo() = default;
        RunInfo(const nlohmann::json& j);
        RunInfo(usize mem_size, float compute_intensity) : mem_size(mem_size), compute_intensity(compute_intensity) {}
        friend auto operator < (const RunInfo& a, const RunInfo& b) -> bool;
        friend auto operator << (std::ostream& os, const RunInfo& info) -> std::ostream&;
    };

    struct Node {
        enum Status {
            WAITING,
            RUNNING,
            FINISHED
        };

        struct KernelInfo {
            std::string kernel_name;
            std::vector<int> targets;
            KernelInfo() = default;
            KernelInfo(const nlohmann::json& j);
            friend auto operator << (std::ostream& os, const KernelInfo& info) -> std::ostream&;
        };

        struct MemInfo {
            MemType type;
            usize size;
            u32 target;
            MemInfo() = default;
            MemInfo(const nlohmann::json& j);
            friend auto operator << (std::ostream& os, const MemInfo& info) -> std::ostream&;
        };

        struct CopyInfo {
            CopyType type;
            usize size;
            u32 src, dst;
            CopyInfo() = default;
            CopyInfo(const nlohmann::json& j);
            friend auto operator << (std::ostream& os, const CopyInfo& info) -> std::ostream&;
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
            u32 logn;
            u32 target;
            NttInfo() = default;
            NttInfo(const nlohmann::json& j);
            friend auto operator << (std::ostream& os, const NttInfo& info) -> std::ostream&;
        };

        std::variant<KernelInfo, MemInfo, CopyInfo, NttInfo> info; // the information of the node
        std::vector<int> last, next; // the last and next nodes
        std::set<int> wait_for; // the nodes that this node is waiting for
        const bool is_cuda; // whether this node is a cuda node
        std::variant<std::future<void>, cudaStream_t> stream; // the stream of the node
        Status status;
        RunInfo run_info;
        u32 id;

        Node(const nlohmann::json& j);
        void sync();
        friend auto operator << (std::ostream& os, const Node& node) -> std::ostream&;
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
            friend auto operator << (std::ostream& os, const ComputeGraph& graph) -> std::ostream&;
    };
}