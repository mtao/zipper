#if !defined(ZIPPER_EXPRESSION_UNARY_TAN_HPP)
#define ZIPPER_EXPRESSION_UNARY_TAN_HPP

#include "CoefficientWiseOperation.hpp"
#include <cmath>

namespace zipper::expression::unary {
namespace detail {
template <typename A> struct tan {
  static constexpr A operator()(const A &a) { return std::tan(a); }
};
} // namespace detail

template <zipper::concepts::Expression Child>
using Tan = CoefficientWiseOperation<
    Child,
    detail::tan<
        typename zipper::expression::detail::ExpressionTraits<Child>::value_type>>;

} // namespace zipper::expression::unary
#endif
