#if !defined(ZIPPER_EXPRESSION_UNARY_SQRT_HPP)
#define ZIPPER_EXPRESSION_UNARY_SQRT_HPP

#include "CoefficientWiseOperation.hpp"
#include <cmath>

namespace zipper::expression::unary {
namespace detail {
template <typename A> struct sqrt {
  static constexpr A operator()(const A &a) { return std::sqrt(a); }
};
} // namespace detail

template <zipper::concepts::Expression Child>
using Sqrt = CoefficientWiseOperation<
    Child,
    detail::sqrt<
        typename zipper::expression::detail::ExpressionTraits<Child>::value_type>>;

} // namespace zipper::expression::unary
#endif
