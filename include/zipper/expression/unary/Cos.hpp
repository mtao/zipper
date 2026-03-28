#if !defined(ZIPPER_EXPRESSION_UNARY_COS_HPP)
#define ZIPPER_EXPRESSION_UNARY_COS_HPP

#include "CoefficientWiseOperation.hpp"
#include <cmath>

namespace zipper::expression::unary {
namespace detail {
template <typename A> struct cos {
  static constexpr A operator()(const A &a) { return std::cos(a); }
};
} // namespace detail

template <zipper::concepts::Expression Child>
using Cos = CoefficientWiseOperation<
    Child,
    detail::cos<
        typename zipper::expression::detail::ExpressionTraits<Child>::value_type>>;

} // namespace zipper::expression::unary
#endif
