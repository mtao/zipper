#if !defined(ZIPPER_EXPRESSION_UNARY_MATH_EXP2_HPP)
#define ZIPPER_EXPRESSION_UNARY_MATH_EXP2_HPP

#include "../CoefficientWiseOperation.hpp"
#include <cmath>

namespace zipper::expression::unary {
namespace detail {
    template <typename A>
    struct exp2 {
        static constexpr A operator()(const A &a) { return std::exp2(a); }
    };
} // namespace detail

template <zipper::concepts::Expression Child>
using Exp2 = CoefficientWiseOperation<
    Child,
    detail::exp2<typename zipper::expression::detail::ExpressionTraits<
        Child>::value_type>>;

} // namespace zipper::expression::unary
#endif
