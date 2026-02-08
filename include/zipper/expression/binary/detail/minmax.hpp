#if !defined(ZIPPER_EXPRESSION_BINARY_DETAIL_MINMAX_HPP)

#define ZIPPER_EXPRESSION_BINARY_DETAIL_MINMAX_HPP

#include <algorithm>

namespace zipper::expression::binary::detail {
template <typename T>
struct min {
    constexpr decltype(auto) operator()(const T& a, const T& b) const {
        return std::min(a, b);
    }
};
template <typename T>
struct max {
    constexpr decltype(auto) operator()(const T& a, const T& b) const {
        return std::max(a, b);
    }
};

}  // namespace zipper::expression::binary::detail
#endif
