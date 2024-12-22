#include "field_type.h"
#include <stdexcept>

namespace runtime {
    auto get_field_type(const std::string &field_type) -> FieldType {
        if (field_type == "bn256_fr") {
            return FieldType::BN254_FR;
        } else if (field_type == "pasta_fp") {
            return FieldType::PASTA_FP;
        } else {
            throw std::invalid_argument("Invalid field type: " + field_type);
        }
    }

    auto operator<<(std::ostream &os, const FieldType &field_type) -> std::ostream & {
        switch (field_type) {
            case FieldType::BN254_FR:
                os << std::string("bn256_fr");
                break;
            case FieldType::PASTA_FP:
                os << std::string("pasta_fp");
                break;
            default:
                throw std::invalid_argument("Invalid field type");
        }
        return os;
    }
}