#if !defined(ZIPPER_EXPRESSION_UNARY_SIGN_HPP)
#define ZIPPER_EXPRESSION_UNARY_SIGN_HPP

#include "CoefficientWiseOperation.hpp"

namespace zipper::expression::unary {
namespace detail {
template <typename A> struct sign {
  static constexpr A operator()(const A &a) {
    return static_cast<A>((a > A{0}) - (a < A{0}));
  }
};
} // namespace detail

template <zipper::concepts::Expression Child>
using Sign = CoefficientWiseOperation<
    Child,
    detail::sign<
        typename zipper::expression::detail::ExpressionTraits<Child>::value_type>>;

} // namespace zipper::expression::unary
#endif
