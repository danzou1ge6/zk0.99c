#include "basic_types.h"
#include "field_type.h"
#include "graph.cuh"
#include "error.cuh"
#include "memory.h"
#include <cassert>
#include <ostream>

namespace runtime {

    RunInfo::RunInfo(const nlohmann::json& j) {
        mem_size = j["gpu_mem"].template get<usize>();
        compute_intensity = j["compute_intensity"].template get<float>();
    }

    auto operator < (const RunInfo& a, const RunInfo& b) -> bool {
        if (a.mem_size == b.mem_size) {
            return a.compute_intensity > b.compute_intensity;
        }
        return a.mem_size > b.mem_size;
    }

    auto operator << (std::ostream& os, const Node::KernelInfo& info) -> std::ostream& {
        os << "KernelInfo(kernel_name=" << info.kernel_name << ", targets=[";
        for (auto& target : info.targets) {
            os << target << ", ";
        }
        os << "])";
        return os;
    }

    auto operator << (std::ostream& os, const Node::MemInfo& info) -> std::ostream& {
        os << "MemInfo(type=" << info.type << ", size=" << info.size << ", target=" << info.target << ")";
        return os;
    }

    auto operator << (std::ostream& os, const Node::CopyInfo& info) -> std::ostream& {
        os << "CopyInfo(type=" << info.type << ", size=" << info.size << ", src=" << info.src << ", dst=" << info.dst << ")";
        return os;
    }

    auto operator << (std::ostream& os, const Node::NttInfo& info) -> std::ostream& {
        os << "NttInfo(type=" << info.type << ", field=" << info.field << ", logn=" << info.logn << ", target=" << info.target << ")";
        return os;
    }

    auto operator << (std::ostream& os, const Node::NttInfo::NttType& type) -> std::ostream& {
        switch (type) {
            case Node::NttInfo::NttType::NTT:
                os << "NTT";
                break;
            case Node::NttInfo::NttType::INVERSE:
                os << "INVERSE";
                break;
            case Node::NttInfo::NttType::COEFF_TO_EXT:
                os << "COEFF_TO_EXT";
                break;
            case Node::NttInfo::NttType::EXT_TO_COEFF:
                os << "EXT_TO_COEFF";
                break;
        }
        return os;
    }

    auto operator << (std::ostream& os, const RunInfo& info) -> std::ostream& {
        os << "RunInfo(mem_size=" << info.mem_size << ", compute_intensity=" << info.compute_intensity << ")";
        return os;
    }

    auto operator << (std::ostream& os, const Node::Status& status) -> std::ostream& {
        switch (status) {
            case Node::Status::WAITING:
                os << "WAITING";
                break;
            case Node::Status::RUNNING:
                os << "RUNNING";
                break;
            case Node::Status::FINISHED:
                os << "FINISHED";
                break;
        }
        return os;
    }

    auto operator << (std::ostream& os, const Node& node) -> std::ostream& {
        os << "Node(id=" << node.id << ", type=";
        if (std::holds_alternative<Node::KernelInfo>(node.info)) {
            os << "kernel, info=" << std::get<Node::KernelInfo>(node.info);
        } else if (std::holds_alternative<Node::MemInfo>(node.info)) {
            os << "mem, info=" << std::get<Node::MemInfo>(node.info);
        } else if (std::holds_alternative<Node::CopyInfo>(node.info)) {
            os << "copy, info=" << std::get<Node::CopyInfo>(node.info);
        } else if (std::holds_alternative<Node::NttInfo>(node.info)) {
            os << "ntt, info=" << std::get<Node::NttInfo>(node.info);
        }
        os << ", is_cuda=" << node.is_cuda << ", last=[";
        for (auto& last : node.last) {
            os << last << ", ";
        }
        os << "], next=[";
        for (auto& next : node.next) {
            os << next << ", ";
        }
        os << "], status=" << node.status;
        os << ", wait_for=[";
        for (auto& wait_for : node.wait_for) {
            os << wait_for << ", ";
        }
        os << "], run_info=" << node.run_info << ")";
        return os;
    }

    auto operator << (std::ostream& os, const ComputeGraph& graph) -> std::ostream& {
        os << "ComputeGraph(nodes=[" << std::endl;
        for (auto& node : graph.nodes) {
            os << node << std::endl;
        }
        os << "]," << std::endl << "available_nodes=[";
        for (auto& available_node : graph.available_nodes) {
            os << available_node.first << " -> " << available_node.second << ", ";
        }
        os << "])";
        return os;
    }

    Node::KernelInfo::KernelInfo(const nlohmann::json& j) {
        kernel_name = j["kernel_name"].template get<std::string>();
        targets = j["targets"].template get<std::vector<int>>();
    }

    Node::MemInfo::MemInfo(const nlohmann::json& j) {
        size = j["size"].template get<size_t>();
        target = j["target"].template get<unsigned int>();
        auto mem_typ_str = j["mem_type"].template get<std::string>();
        type = get_mem_type(mem_typ_str);
    }

    Node::CopyInfo::CopyInfo(const nlohmann::json& j) {
        size = j["size"].template get<size_t>();
        auto type_str = j["copy_type"].template get<std::string>();
        if (type_str == "host_to_device") {
            type = CopyType::H2D;
        } else if (type_str == "device_to_host") {
            type = CopyType::D2H;
        } else if (type_str == "device_to_device") {
            type = CopyType::D2D;
        } else if (type_str == "host_to_host") {
            type = CopyType::H2H;
        } else {
            throw std::runtime_error("Unknown copy type");
        }
        src = j["src"].template get<u32>();
        dst = j["dst"].template get<u32>();
    }

    Node::NttInfo::NttInfo(const nlohmann::json& j) {
        logn = j["logn"].template get<unsigned int>();
        auto type_str = j["ntt_type"].template get<std::string>();
        if (type_str == "ntt") {
            type = NttType::NTT;
        } else if (type_str == "inverse") {
            type = NttType::INVERSE;
        } else if (type_str == "coeff_to_ext") {
            type = NttType::COEFF_TO_EXT;
        } else if (type_str == "ext_to_coeff") {
            type = NttType::EXT_TO_COEFF;
        } else {
            throw std::runtime_error("Unknown NTT type");
        }
        auto field_str = j["field"].template get<std::string>();
        field = get_field_type(field_str);
        target = j["target"].template get<u32>();
    }

    Node::Node(const nlohmann::json& j)
    : is_cuda(j["is_cuda"]), last(j["last"].template get<std::vector<int>>()), run_info(j), id(j["id"]),
    next(j["next"].template get<std::vector<int>>()), wait_for(last.begin(), last.end())
    , status(Node::Status::WAITING) {
        if (j["type"] == "kernel") {
            info = KernelInfo(j);
        } else if (j["type"] == "mem") {
            info = MemInfo(j);
        } else if (j["type"] == "copy") {
            info = CopyInfo(j);
        } else if (j["type"] == "ntt") {
            info = NttInfo(j);
        }
    }

    auto Node::sync() -> void {
        if (is_cuda) {
            CUDA_CHECK_EXIT(cudaStreamSynchronize(*std::get_if<cudaStream_t>(&stream)));
        } else {
            std::get_if<std::future<void>>(&stream)->get();
        }
    }

    ComputeGraph::ComputeGraph(const nlohmann::json& j) {
        nodes.reserve(j.size());
        for (auto &node : j) {
            assert(node["id"] == nodes.size());
            nodes.push_back(Node(node));
        }
        available_nodes.insert(std::make_pair(nodes[0].run_info, 0));
    }

    auto ComputeGraph::get_next() -> int const {
        auto it = available_nodes.begin();
        if (it == available_nodes.end()) {
            throw std::runtime_error("No available nodes");
        }
        return it->second;
    }

    auto ComputeGraph::get_next(int memory, float compute_intensity) -> int const {
        if (available_nodes.empty()) {
            throw std::runtime_error("No available nodes");
        }
        auto it = available_nodes.lower_bound(RunInfo(memory, compute_intensity));
        if (it == available_nodes.end()) {
            it--;
            if (it->first.mem_size > memory) {
                throw std::runtime_error("Out of memory");
            }
        }
        return it->second;
    }

    auto ComputeGraph::run_node(int id) -> void {
        if (nodes[id].last.size() == 1 &&
        nodes[id].is_cuda == true &&
        nodes[nodes[id].last[0]].is_cuda == true) {
            nodes[id].stream = *std::get_if<cudaStream_t>(&nodes[nodes[id].last[0]].stream);
            nodes[nodes[id].last[0]].status = Node::Status::FINISHED;
        } else {
            for (auto& last : nodes[id].last) {
                nodes[last].sync();
                nodes[last].status = Node::Status::FINISHED;
            }
            if (nodes[id].is_cuda) {
                CUDA_CHECK_EXIT(cudaStreamCreate(std::get_if<cudaStream_t>(&nodes[id].stream)));
            }
        }
        // TODO : invoke the runner


        nodes[id].status = Node::Status::RUNNING;
        for (auto& next : nodes[id].next) {
            nodes[next].wait_for.erase(id);
            if (nodes[next].wait_for.empty()) {
                available_nodes.insert(std::make_pair(nodes[next].run_info, next));
            }
        }
    }
    auto ComputeGraph::finish() -> bool const {
        return available_nodes.empty();
    }
    
}