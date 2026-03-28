#if !defined(ZIPPER_EXPRESSION_UNARY_ASIN_HPP)
#define ZIPPER_EXPRESSION_UNARY_ASIN_HPP

#include "CoefficientWiseOperation.hpp"
#include <cmath>

namespace zipper::expression::unary {
namespace detail {
template <typename A> struct asin {
  static constexpr A operator()(const A &a) { return std::asin(a); }
};
} // namespace detail

template <zipper::concepts::Expression Child>
using Asin = CoefficientWiseOperation<
    Child,
    detail::asin<
        typename zipper::expression::detail::ExpressionTraits<Child>::value_type>>;

} // namespace zipper::expression::unary
#endif
