#if !defined(ZIPPER_EXPRESSION_UNARY_MATH_EXP_HPP)
#define ZIPPER_EXPRESSION_UNARY_MATH_EXP_HPP

#include "../CoefficientWiseOperation.hpp"
#include <cmath>

namespace zipper::expression::unary {
namespace detail {
    template <typename A>
    struct exp {
        static constexpr A operator()(const A &a) { return std::exp(a); }
    };
} // namespace detail

template <zipper::concepts::Expression Child>
using Exp = CoefficientWiseOperation<
    Child,
    detail::exp<typename zipper::expression::detail::ExpressionTraits<
        Child>::value_type>>;

} // namespace zipper::expression::unary
#endif
