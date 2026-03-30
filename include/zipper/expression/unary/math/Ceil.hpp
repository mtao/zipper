#if !defined(ZIPPER_EXPRESSION_UNARY_MATH_CEIL_HPP)
#define ZIPPER_EXPRESSION_UNARY_MATH_CEIL_HPP

#include "../CoefficientWiseOperation.hpp"
#include "../ZeroPreserving.hpp"
#include <cmath>

namespace zipper::expression::unary {
namespace detail {
    template <typename A>
    struct ceil {
        static constexpr A operator()(const A &a) { return std::ceil(a); }
    };
} // namespace detail

template <zipper::concepts::Expression Child>
using Ceil = CoefficientWiseOperation<
    Child,
    detail::ceil<typename zipper::expression::detail::ExpressionTraits<
        Child>::value_type>>;

} // namespace zipper::expression::unary

template <typename T>
struct zipper::expression::detail::is_zero_preserving_unary_op<
    zipper::expression::unary::detail::ceil<T>> : std::true_type {};

#endif
