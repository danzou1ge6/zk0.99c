#pragma once

#include "self_sort_in_place_ntt.cuh"
#include "../../wrapper/ntt/c_api/ntt_c_api.h"
#include <cstddef>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <memory>
#include <deque>

// pasta_fp
// 28948022309329048855892746252171976963363056481941560715954676764349967630337
const auto params_pasta_fp = mont256::Params {
  .m = BIG_INTEGER_CHUNKS8(0x40000000, 0x00000000, 0x00000000, 0x00000000, 0x224698fc, 0x094cf91b, 0x992d30ed, 0x00000001),
  .r_mod = BIG_INTEGER_CHUNKS8(0x3fffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x992c350b, 0xe41914ad, 0x34786d38, 0xfffffffd),
  .r2_mod = BIG_INTEGER_CHUNKS8(0x96d41af, 0x7b9cb714, 0x7797a99b, 0xc3c95d18, 0xd7d30dbd, 0x8b0de0e7, 0x8c78ecb3, 0x0000000f),
  .m_prime = 4294967295
};

// bn256_fr
// 21888242871839275222246405745257275088548364400416034343698204186575808495617
const auto params_bn256_fr = mont256::Params {
  .m = BIG_INTEGER_CHUNKS8(0x30644e72, 0xe131a029, 0xb85045b6, 0x8181585d, 0x2833e848, 0x79b97091, 0x43e1f593, 0xf0000001),
  .r_mod = BIG_INTEGER_CHUNKS8(0xe0a77c1, 0x9a07df2f, 0x666ea36f, 0x7879462e, 0x36fc7695, 0x9f60cd29, 0xac96341c, 0x4ffffffb),
  .r2_mod = BIG_INTEGER_CHUNKS8(0x216d0b1, 0x7f4e44a5, 0x8c49833d, 0x53bb8085, 0x53fe3ab1, 0xe35c59e3, 0x1bb8e645, 0xae216da7),
  .m_prime = 4026531839     
};

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
                        ntt_kernel = std::make_shared<ntt::self_sort_in_place_ntt<8> >(params_pasta_fp, omega, id.log_len, false, avail / instance_memory, id.inverse, id.process, inv_n, zeta);
                    } else if (id.field == FIELD::HALO2CURVES_BN256_FR) {
                        cudaMemGetInfo(&avail, &total);
                        auto instance_memory = 8ull * sizeof(uint) * (1 << id.log_len);
                        avail -= instance_memory / 2;
                        ntt_kernel = std::make_shared<ntt::self_sort_in_place_ntt<8> >(params_bn256_fr, omega, id.log_len, false, avail / instance_memory, id.inverse, id.process, inv_n, zeta);
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