#if !defined(ZIPPER_EXPRESSION_UNARY_MATH_SINH_HPP)
#define ZIPPER_EXPRESSION_UNARY_MATH_SINH_HPP

#include "../CoefficientWiseOperation.hpp"
#include "../detail/ZeroPreserving.hpp"
#include <cmath>

namespace zipper::expression::unary {
namespace detail {
    template <typename A>
    struct sinh {
        static constexpr A operator()(const A &a) { return std::sinh(a); }
    };
} // namespace detail

template <zipper::concepts::Expression Child>
using Sinh = CoefficientWiseOperation<
    Child,
    detail::sinh<typename zipper::expression::detail::ExpressionTraits<
        Child>::value_type>>;

} // namespace zipper::expression::unary

template <typename T>
struct zipper::expression::detail::is_zero_preserving_unary_op<
    zipper::expression::unary::detail::sinh<T>> : std::true_type {};

#endif
