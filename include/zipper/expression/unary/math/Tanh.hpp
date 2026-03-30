#if !defined(ZIPPER_EXPRESSION_UNARY_MATH_TANH_HPP)
#define ZIPPER_EXPRESSION_UNARY_MATH_TANH_HPP

#include "../CoefficientWiseOperation.hpp"
#include "../detail/ZeroPreserving.hpp"
#include <cmath>

namespace zipper::expression::unary {
namespace detail {
    template <typename A>
    struct tanh {
        static constexpr A operator()(const A &a) { return std::tanh(a); }
    };
} // namespace detail

template <zipper::concepts::Expression Child>
using Tanh = CoefficientWiseOperation<
    Child,
    detail::tanh<typename zipper::expression::detail::ExpressionTraits<
        Child>::value_type>>;

} // namespace zipper::expression::unary

template <typename T>
struct zipper::expression::detail::is_zero_preserving_unary_op<
    zipper::expression::unary::detail::tanh<T>> : std::true_type {};

#endif
