#if !defined(ZIPPER_EXPRESSION_BINARY_DETAIL_MINMAX_HPP)

#define ZIPPER_EXPRESSION_BINARY_DETAIL_MINMAX_HPP

#include <algorithm>

namespace zipper::expression::binary::detail {
template <typename T>
struct min {
    static constexpr decltype(auto) operator()(const T& a, const T& b) {
        return std::min(a, b);
    }
};
template <typename T>
struct max {
    static constexpr decltype(auto) operator()(const T& a, const T& b) {
        return std::max(a, b);
    }
};

}  // namespace zipper::expression::binary::detail
#endif
