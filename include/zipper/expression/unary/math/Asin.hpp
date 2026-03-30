#if !defined(ZIPPER_EXPRESSION_UNARY_MATH_ASIN_HPP)
#define ZIPPER_EXPRESSION_UNARY_MATH_ASIN_HPP

#include "../CoefficientWiseOperation.hpp"
#include "../ZeroPreserving.hpp"
#include <cmath>

namespace zipper::expression::unary {
namespace detail {
    template <typename A>
    struct asin {
        static constexpr A operator()(const A &a) { return std::asin(a); }
    };
} // namespace detail

template <zipper::concepts::Expression Child>
using Asin = CoefficientWiseOperation<
    Child,
    detail::asin<typename zipper::expression::detail::ExpressionTraits<
        Child>::value_type>>;

} // namespace zipper::expression::unary

template <typename T>
struct zipper::expression::detail::is_zero_preserving_unary_op<
    zipper::expression::unary::detail::asin<T>> : std::true_type {};

#endif
