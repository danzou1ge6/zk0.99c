#pragma once
#include <string>

namespace runtime {
    enum FieldType {
        BN256_FR,
        PASTA_FP,
    };

    auto get_field_type(const std::string &field_type) -> FieldType;

    auto operator<<(std::ostream &os, const FieldType &field_type) -> std::ostream &;
}