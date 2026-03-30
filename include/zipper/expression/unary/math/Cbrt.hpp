#if !defined(ZIPPER_EXPRESSION_UNARY_MATH_CBRT_HPP)
#define ZIPPER_EXPRESSION_UNARY_MATH_CBRT_HPP

#include "../CoefficientWiseOperation.hpp"
#include "../ZeroPreserving.hpp"
#include <cmath>

namespace zipper::expression::unary {
namespace detail {
    template <typename A>
    struct cbrt {
        static constexpr A operator()(const A &a) { return std::cbrt(a); }
    };
} // namespace detail

template <zipper::concepts::Expression Child>
using Cbrt = CoefficientWiseOperation<
    Child,
    detail::cbrt<typename zipper::expression::detail::ExpressionTraits<
        Child>::value_type>>;

} // namespace zipper::expression::unary

template <typename T>
struct zipper::expression::detail::is_zero_preserving_unary_op<
    zipper::expression::unary::detail::cbrt<T>> : std::true_type {};

#endif
