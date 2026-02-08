#if !defined(ZIPPER_UTILS_EXTENTS_EXTENT_ARITHMETIC_HPP)
#define ZIPPER_UTILS_EXTENTS_EXTENT_ARITHMETIC_HPP

#include "zipper/concepts/Index.hpp"
#include "zipper/types.hpp"

namespace zipper::utils::extents {

template <typename BinOp>
constexpr index_type apply_binop(const index_type a, const index_type b) {
    if (a == std::dynamic_extent || b == std::dynamic_extent) {
        return std::dynamic_extent;
    } else {
        return BinOp{}(a, b);
    }
}

constexpr index_type plus(const index_type a, const index_type b) {
    return apply_binop<std::plus<index_type>>(a, b);
}
constexpr index_type minus(const index_type a, const index_type b) {
    return apply_binop<std::minus<index_type>>(a, b);
}

constexpr index_type offset(const index_type a, const int64_t b) {
    if (b >= 0) {
        return plus(a, index_type(b));
    } else {
        return minus(a, index_type(-b));
    }
}
constexpr index_type multiplies(const index_type a, const index_type b) {
    return apply_binop<std::multiplies<index_type>>(a, b);
}
constexpr index_type divides(const index_type a, const index_type b) {
    return apply_binop<std::divides<index_type>>(a, b);
}
constexpr index_type modulus(const index_type a, const index_type b) {
    return apply_binop<std::modulus<index_type>>(a, b);
}

// constexpr index_type min(const index_type a, const index_type b) {
//     return apply_binop<std::min<index_type>>(a, b);
// }
//
// constexpr index_type max(const index_type a, const index_type b) {
//     return apply_binop<std::max<index_type>>(a, b);
// }

}  // namespace zipper::utils::extents
#endif
