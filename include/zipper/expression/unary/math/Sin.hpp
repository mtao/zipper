#if !defined(ZIPPER_EXPRESSION_UNARY_MATH_SIN_HPP)
#define ZIPPER_EXPRESSION_UNARY_MATH_SIN_HPP

#include "../CoefficientWiseOperation.hpp"
#include "../ZeroPreserving.hpp"
#include <cmath>

namespace zipper::expression::unary {
namespace detail {
    template <typename A>
    struct sin {
        static constexpr A operator()(const A &a) { return std::sin(a); }
    };
} // namespace detail

template <zipper::concepts::Expression Child>
using Sin = CoefficientWiseOperation<
    Child,
    detail::sin<typename zipper::expression::detail::ExpressionTraits<
        Child>::value_type>>;

} // namespace zipper::expression::unary

template <typename T>
struct zipper::expression::detail::is_zero_preserving_unary_op<
    zipper::expression::unary::detail::sin<T>> : std::true_type {};

#endif
