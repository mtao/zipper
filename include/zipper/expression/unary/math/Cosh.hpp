#if !defined(ZIPPER_EXPRESSION_UNARY_MATH_COSH_HPP)
#define ZIPPER_EXPRESSION_UNARY_MATH_COSH_HPP

#include "../CoefficientWiseOperation.hpp"
#include <cmath>

namespace zipper::expression::unary {
namespace detail {
    template <typename A>
    struct cosh {
        static constexpr A operator()(const A &a) { return std::cosh(a); }
    };
} // namespace detail

template <zipper::concepts::Expression Child>
using Cosh = CoefficientWiseOperation<
    Child,
    detail::cosh<typename zipper::expression::detail::ExpressionTraits<
        Child>::value_type>>;

} // namespace zipper::expression::unary
#endif
