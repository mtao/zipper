#if !defined(ZIPPER_EXPRESSION_UNARY_SCALARPOWER_HPP)
#define ZIPPER_EXPRESSION_UNARY_SCALARPOWER_HPP

#include <cmath>

#include "ScalarOperation.hpp"

namespace zipper::expression::unary {
namespace detail {

template <typename T> struct pow {
  constexpr auto operator()(const T &a, const T &b) const {
    return std::pow(a, b);
  }
};
} // namespace detail

template <zipper::concepts::Expression Child, typename Scalar>
using ScalarPower =
    ScalarOperation<Child, detail::pow<Scalar>, Scalar, true>;

} // namespace zipper::expression::unary

#endif
