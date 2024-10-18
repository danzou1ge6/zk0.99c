#include "graph.cuh"
#include "error.cuh"

namespace runtime {

    RunInfo::RunInfo(const nlohmann::json& j) {
        mem_size = j["gpu_mem"].template get<int>();
        compute_intensity = j["compute_intensity"].template get<float>();
    }

    auto operator < (const RunInfo& a, const RunInfo& b) -> bool {
        if (a.mem_size == b.mem_size) {
            return a.compute_intensity > b.compute_intensity;
        }
        return a.mem_size > b.mem_size;
    }

    Node::KernelInfo::KernelInfo(const nlohmann::json& j) {
        kernel_name = j["kernel_name"].template get<std::string>();
        input_num = j["input_num"].template get<int>();
        inputs = j["inputs"].template get<std::vector<int>>();
    }

    Node::MemInfo::MemInfo(const nlohmann::json& j) {
        size = j["size"].template get<size_t>();
        register_id = j["register_id"].template get<unsigned int>();
        auto mem_typ_str = j["type"].template get<std::string>();
        if (mem_typ_str == "host") {
            type = MemType::HOST;
        } else if (mem_typ_str == "device") {
            type = MemType::DEVICE;
        } else if (mem_typ_str == "managed") {
            type = MemType::MANAGED;
        } else if (mem_typ_str == "pinned") {
            type = MemType::PINNED;
        } else {
            throw std::runtime_error("Unknown memory type");
        }
    }

    Node::CopyInfo::CopyInfo(const nlohmann::json& j) {
        size = j["size"].template get<size_t>();
        auto type_str = j["type"].template get<std::string>();
        if (type_str == "host_to_device") {
            type = cudaMemcpyHostToDevice;
        } else if (type_str == "device_to_host") {
            type = cudaMemcpyDeviceToHost;
        } else if (type_str == "device_to_device") {
            type = cudaMemcpyDeviceToDevice;
        } else if (type_str == "host_to_host") {
            type = cudaMemcpyHostToHost;
        } else {
            throw std::runtime_error("Unknown copy type");
        }
    }

    Node::NttInfo::NttInfo(const nlohmann::json& j) {
        // TODO
    }

    Node::Node(const nlohmann::json& j)
    : is_cuda(j["is_cuda"]), last(j["last"].template get<std::vector<int>>()), run_info(j),
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
        // TODO
    }

    ComputeGraph::ComputeGraph(const nlohmann::json& j) {
        nodes.reserve(j.size());
        for (auto &node : j) {
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
        // TODO
        return 0;
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