#if !defined(ZIPPER_EXPRESSION_BINARY_DETAIL_MINMAX_HPP)

#define ZIPPER_EXPRESSION_BINARY_DETAIL_MINMAX_HPP

#include <algorithm>

namespace zipper::expression::binary::detail {
template <typename T>
struct min {
    // By value: arguments may be temporary coefficients.
    static constexpr auto operator()(const T& a, const T& b) -> T {
        return std::min(a, b);
    }
};
template <typename T>
struct max {
    // By value: arguments may be temporary coefficients.
    static constexpr auto operator()(const T& a, const T& b) -> T {
        return std::max(a, b);
    }
};

}  // namespace zipper::expression::binary::detail
#endif
