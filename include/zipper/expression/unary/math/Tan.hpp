#if !defined(ZIPPER_EXPRESSION_UNARY_MATH_TAN_HPP)
#define ZIPPER_EXPRESSION_UNARY_MATH_TAN_HPP

#include "../CoefficientWiseOperation.hpp"
#include "../ZeroPreserving.hpp"
#include <cmath>

namespace zipper::expression::unary {
namespace detail {
    template <typename A>
    struct tan {
        static constexpr A operator()(const A &a) { return std::tan(a); }
    };
} // namespace detail

template <zipper::concepts::Expression Child>
using Tan = CoefficientWiseOperation<
    Child,
    detail::tan<typename zipper::expression::detail::ExpressionTraits<
        Child>::value_type>>;

} // namespace zipper::expression::unary

template <typename T>
struct zipper::expression::detail::is_zero_preserving_unary_op<
    zipper::expression::unary::detail::tan<T>> : std::true_type {};

#endif
