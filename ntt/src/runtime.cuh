#pragma once

#include "self_sort_in_place_ntt.cuh"
#include "../../wrapper/ntt/c_api/ntt_c_api.h"
#include "../../mont/src/bn256_fr.cuh"
#include "../../mont/src/pasta_fp.cuh"
#include <cstddef>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <memory>
#include <deque>

namespace runtime {
    struct ntt_id {
        uint log_len;
        FIELD field;
        bool process, inverse;
        friend bool operator < (const ntt_id &a, const ntt_id &b) {
            if (a.log_len != b.log_len) return a.log_len < b.log_len;
            if (a.field != b.field) return a.field < b.field;
            if (a.process != b.process) return a.process < b.process;
            if (a.inverse != b.inverse) return a.inverse < b.inverse;
            return false;
        }
        friend bool operator == (const ntt_id &a, const ntt_id &b) {
            if (a.log_len != b.log_len) return false;
            if (a.field != b.field) return false;
            if (a.process != b.process) return false;
            if (a.inverse != b.inverse) return false;
            return true;
        }
        friend bool operator != (const ntt_id &a, const ntt_id &b) {
            return !(a == b);
        }
    };

    class fifo {
        std::deque<ntt_id> q;
        std::shared_mutex mtx;
        std::shared_ptr<std::map<ntt_id, std::shared_ptr<ntt::best_ntt> > > ntt_kernels;
        std::shared_ptr<std::shared_mutex> mtx_ntt_kernels; // lock for map
        uint max_num;
        public:
        fifo(std::shared_ptr<std::map<ntt_id, std::shared_ptr<ntt::best_ntt> > > ntt_kernels, std::shared_ptr<std::shared_mutex> mtx_ntt_kernels, uint max_num = 1) : ntt_kernels(ntt_kernels), mtx_ntt_kernels(mtx_ntt_kernels), max_num(max_num) {}
        void set_num(uint num) {
            std::unique_lock<std::shared_mutex> wlock(mtx);
            if (num < max_num) {
                std::unique_lock<std::shared_mutex> wlock_map(*mtx_ntt_kernels);
                while (q.size() >= max_num) {
                    auto old = q.front();
                    q.pop_front();
                    (*ntt_kernels)[old]->clean_gpu();                
                }
            }
            max_num = num;
        }
        void arrive(ntt_id id) {
            std::shared_lock<std::shared_mutex> rlock(mtx);
            for (auto cur_id : q) {
                if (cur_id == id) return;
            }
            rlock.unlock();
            std::unique_lock<std::shared_mutex> wlock(mtx);
            for (auto cur_id : q) {
                if (cur_id == id) return;
            }
            std::unique_lock<std::shared_mutex> wlock_map(*mtx_ntt_kernels);
            if (q.size() >= max_num) {
                auto old = q.front();
                q.pop_front();
                (*ntt_kernels)[old]->clean_gpu();                
            }
            q.push_back(id);
            (*ntt_kernels)[id]->to_gpu();
        }
        void clean_all() {
            std::unique_lock<std::shared_mutex> wlock(mtx);
            std::unique_lock<std::shared_mutex> wlock_map(*mtx_ntt_kernels);
            while (!q.empty()) {
                auto old = q.front();
                q.pop_front();
                (*ntt_kernels)[old]->clean_gpu();                
            }
        }
    };

    template <typename T>
    class ntt_runtime {
        std::shared_ptr<std::map<ntt_id, std::shared_ptr<ntt::best_ntt> > > ntt_kernels;
        std::shared_ptr<std::shared_mutex> mtx_ntt_kernels; // lock for map
        T cache_manager;

        public:
        ntt_runtime(int max_num = 1) : ntt_kernels(std::make_shared<std::map<ntt_id, std::shared_ptr<ntt::best_ntt> > >()),
        mtx_ntt_kernels(std::make_shared<std::shared_mutex>()), cache_manager(ntt_kernels, mtx_ntt_kernels, max_num) {}
        std::shared_ptr<ntt::best_ntt> get_ntt_kernel(ntt_id id, const uint *omega, const uint *inv_n, const uint *zeta) {
            std::shared_lock<std::shared_mutex> rlock(*mtx_ntt_kernels);
            auto iter = ntt_kernels->find(id);
            rlock.unlock();
            if (iter == ntt_kernels->end()) {
                std::unique_lock<std::shared_mutex> wlock(*mtx_ntt_kernels);
                iter = ntt_kernels->find(id);
                if (iter == ntt_kernels->end()) {
                    std::shared_ptr<ntt::best_ntt> ntt_kernel;
                    size_t avail, total;
                    if (id.field == FIELD::PASTA_CURVES_FIELDS_FP) {
                        cudaMemGetInfo(&avail, &total);
                        auto instance_memory = 8ull * sizeof(uint) * (1 << id.log_len);
                        avail -= instance_memory / 2;
                        ntt_kernel = std::make_shared<ntt::self_sort_in_place_ntt<pasta_fp::Element> >(omega, id.log_len, false, avail / instance_memory, id.inverse, id.process, inv_n, zeta);
                    } else if (id.field == FIELD::HALO2CURVES_BN256_FR) {
                        cudaMemGetInfo(&avail, &total);
                        auto instance_memory = 8ull * sizeof(uint) * (1 << id.log_len);
                        avail -= instance_memory / 2;
                        ntt_kernel = std::make_shared<ntt::self_sort_in_place_ntt<bn256_fr::Element> >(omega, id.log_len, false, avail / instance_memory, id.inverse, id.process, inv_n, zeta);
                    }
                    ntt_kernels->insert(std::make_pair(id, ntt_kernel));
                    wlock.unlock();
                    cache_manager.arrive(id);
                    return ntt_kernel;
                }
            }
            cache_manager.arrive(id);
            return iter->second;
        }
    };
    
}