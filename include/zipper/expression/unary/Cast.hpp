#if !defined(ZIPPER_EXPRESSION_UNARY_CAST_HPP)
#define ZIPPER_EXPRESSION_UNARY_CAST_HPP

#include "CoefficientWiseOperation.hpp"

namespace zipper::expression::unary {
namespace detail {
template <typename A, typename B> struct cast {
  constexpr static auto operator()(const A &a) { return static_cast<B>(a); }
};
} // namespace detail

// converts an underlying view to a new type
template <typename A, zipper::concepts::Expression Child>
using Cast = CoefficientWiseOperation<
    const Child, detail::cast<typename zipper::expression::detail::
                                  ExpressionTraits<Child>::value_type,
                              A>>;

// converts b into a view of type A
template <typename A, zipper::concepts::Expression B> auto cast(B &b) {
  return Cast<A, B>(b);
}

} // namespace zipper::expression::unary
#endif
