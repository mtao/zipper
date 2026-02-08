#if !defined(ZIPPER_EXPRESSION_UNARY_ABSVIEW_HPP)
#define ZIPPER_EXPRESSION_UNARY_ABSVIEW_HPP

#include "CoefficientWiseOperation.hpp"
#include <cmath>

namespace zipper::expression::unary {
namespace detail {
template <typename A> struct abs {
  constexpr A operator()(const A &a) const { return std::abs(a); }
};
} // namespace detail

template <zipper::concepts::Expression Child>
using Abs = CoefficientWiseOperation<
    Child,
    detail::abs<
        typename zipper::expression::detail::ExpressionTraits<Child>::value_type>>;

} // namespace zipper::expression::unary
#endif
