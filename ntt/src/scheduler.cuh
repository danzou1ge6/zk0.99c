#pragma once

#include "naive_ntt.cuh"
#include "bellperson_ntt.cuh"
#include "self_sort_in_place_ntt.cuh"
#include "../../wrapper/ntt/c_api/ntt_c_api.h"
#include <map>
#include <vector>
#include <memory>

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

namespace scheduler {
    struct ntt_id {
        uint log_len;
        FIELD field;
        std::vector<uint> omega;
        friend bool operator < (const ntt_id &a, const ntt_id &b) {
            if (a.log_len != b.log_len) return a.log_len < b.log_len;
            if (a.field != b.field) return a.field < b.field;
            if (a.omega.size() != b.omega.size()) return a.omega.size() < b.omega.size();
            for (size_t i = 0; i < a.omega.size(); i++) {
                if (a.omega[i] != b.omega[i]) return a.omega[i] < b.omega[i];
            }
            return false;
        }
        friend bool operator == (const ntt_id &a, const ntt_id &b) {
            if (a.log_len != b.log_len) return false;
            if (a.field != b.field) return false;
            if (a.omega.size() != b.omega.size()) return false;
            for (auto i = 0; i < a.omega.size(); i++) {
                if (a.omega[i] != b.omega[i]) return false;
            }
            return true;
        }
        friend bool operator != (const ntt_id &a, const ntt_id &b) {
            return !(a == b);
        }
    };
    std::map<ntt_id, std::shared_ptr<ntt::best_ntt> > ntt_kernels;

    std::shared_ptr<ntt::best_ntt> get_ntt_kernel(ntt_id id) {
        static auto last_id = ntt_id();
        if (last_id != id && last_id.omega.size() != 0) {
            ntt_kernels[last_id]->clean_gpu();
        }
        last_id = id;

        if (ntt_kernels.find(id) == ntt_kernels.end()) {
            std::shared_ptr<ntt::best_ntt> ntt_kernel;
            if (id.field == FIELD::PASTA_CURVES_FIELDS_FP) {
                uint omega[8];
                for (int i = 0; i < 8; i++) omega[i] = id.omega[i];
                ntt_kernel = std::make_shared<ntt::self_sort_in_place_ntt<8> >(params_pasta_fp, omega, id.log_len, false);
            } else if (id.field == FIELD::HALO2CURVES_BN256_FR) {
                uint omega[8];
                for (int i = 0; i < 8; i++) omega[i] = id.omega[i];
                ntt_kernel = std::make_shared<ntt::self_sort_in_place_ntt<8> >(params_bn256_fr, omega, id.log_len, false);
            }
            ntt_kernels[id] = ntt_kernel;
        }
        
        return ntt_kernels[id];
    }
}