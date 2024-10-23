#include "field_type.h"
#include <stdexcept>

namespace runtime {
    auto get_field_type(const std::string &field_type) -> FieldType {
        if (field_type == "bn256_fr") {
            return FieldType::BN256_FR;
        } else if (field_type == "pasta_fp") {
            return FieldType::PASTA_FP;
        } else {
            throw std::invalid_argument("Invalid field type: " + field_type);
        }
    }
}