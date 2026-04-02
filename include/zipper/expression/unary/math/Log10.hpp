#if !defined(ZIPPER_EXPRESSION_UNARY_MATH_LOG10_HPP)
#define ZIPPER_EXPRESSION_UNARY_MATH_LOG10_HPP

#include "../CoefficientWiseOperation.hpp"
#include <cmath>

namespace zipper::expression::unary {
namespace detail {
    template <typename A>
    struct log10 {
        static constexpr A operator()(const A &a) { return std::log10(a); }
    };
} // namespace detail

template <zipper::concepts::Expression Child>
using Log10 = CoefficientWiseOperation<
    Child,
    detail::log10<typename zipper::expression::detail::ExpressionTraits<
        Child>::value_type>>;

} // namespace zipper::expression::unary
#endif
