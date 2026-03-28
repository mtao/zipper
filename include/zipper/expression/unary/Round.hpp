#if !defined(ZIPPER_EXPRESSION_UNARY_ROUND_HPP)
#define ZIPPER_EXPRESSION_UNARY_ROUND_HPP

#include "CoefficientWiseOperation.hpp"
#include <cmath>

namespace zipper::expression::unary {
namespace detail {
template <typename A> struct round {
  static constexpr A operator()(const A &a) { return std::round(a); }
};
} // namespace detail

template <zipper::concepts::Expression Child>
using Round = CoefficientWiseOperation<
    Child,
    detail::round<
        typename zipper::expression::detail::ExpressionTraits<Child>::value_type>>;

} // namespace zipper::expression::unary
#endif
