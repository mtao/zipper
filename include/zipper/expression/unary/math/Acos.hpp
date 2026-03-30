#if !defined(ZIPPER_EXPRESSION_UNARY_MATH_ACOS_HPP)
#define ZIPPER_EXPRESSION_UNARY_MATH_ACOS_HPP

#include "../CoefficientWiseOperation.hpp"
#include <cmath>

namespace zipper::expression::unary {
namespace detail {
    template <typename A>
    struct acos {
        static constexpr A operator()(const A &a) { return std::acos(a); }
    };
} // namespace detail

template <zipper::concepts::Expression Child>
using Acos = CoefficientWiseOperation<
    Child,
    detail::acos<typename zipper::expression::detail::ExpressionTraits<
        Child>::value_type>>;

} // namespace zipper::expression::unary
#endif
