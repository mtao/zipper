#if !defined(ZIPPER_EXPRESSION_UNARY_SINH_HPP)
#define ZIPPER_EXPRESSION_UNARY_SINH_HPP

#include "CoefficientWiseOperation.hpp"
#include <cmath>

namespace zipper::expression::unary {
namespace detail {
template <typename A> struct sinh {
  static constexpr A operator()(const A &a) { return std::sinh(a); }
};
} // namespace detail

template <zipper::concepts::Expression Child>
using Sinh = CoefficientWiseOperation<
    Child,
    detail::sinh<
        typename zipper::expression::detail::ExpressionTraits<Child>::value_type>>;

} // namespace zipper::expression::unary
#endif
