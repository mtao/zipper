#if !defined(ZIPPER_EXPRESSION_UNARY_MATH_LOG2_HPP)
#define ZIPPER_EXPRESSION_UNARY_MATH_LOG2_HPP

#include "../CoefficientWiseOperation.hpp"
#include <cmath>

namespace zipper::expression::unary {
namespace detail {
    template <typename A>
    struct log2 {
        static constexpr A operator()(const A &a) { return std::log2(a); }
    };
} // namespace detail

template <zipper::concepts::Expression Child>
using Log2 = CoefficientWiseOperation<
    Child,
    detail::log2<typename zipper::expression::detail::ExpressionTraits<
        Child>::value_type>>;

} // namespace zipper::expression::unary
#endif
