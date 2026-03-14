/// @file dot.hpp
/// @brief Free-function dot product that bypasses the Form.hpp dependency.
///
/// `VectorBase::dot()` is implemented as `as_form() * o`, which requires
/// `FormBase.hxx` (pulled in via `<zipper/Form.hpp>`) to be included so that
/// the `operator*(FormBase, VectorBase)` overload is visible.  This is a
/// heavyweight include for a simple inner product.
///
/// This header provides `zipper::utils::detail::dot(a, b)` which constructs
/// the same `expression::binary::FormTensorProduct` expression template
/// directly and evaluates its scalar coefficient.  The result is identical to
/// `a.dot(b)` but only requires `FormTensorProduct.hpp` — a much lighter
/// dependency that does not drag in `FormBase` or `Form`.

#if !defined(ZIPPER_UTILS_DETAIL_DOT_HPP)
#define ZIPPER_UTILS_DETAIL_DOT_HPP

#include <type_traits>

#include <zipper/concepts/Vector.hpp>
#include <zipper/expression/binary/FormTensorProduct.hpp>

namespace zipper::utils::detail {

/// @brief Compute the inner product of two vectors without requiring Form.hpp.
///
/// Constructs a `FormTensorProduct` expression from the two vector
/// expressions and evaluates it.  For rank-1 inputs this contracts the single
/// pair of indices, producing a scalar (rank-0) result — exactly the same
/// computation as `VectorBase::dot()`.
///
/// @param a  First vector (any type satisfying `concepts::Vector`).
/// @param b  Second vector (any type satisfying `concepts::Vector`).
/// @return   The dot product as a scalar value_type.
template <concepts::Vector A, concepts::Vector B>
auto dot(const A &a, const B &b) -> typename std::decay_t<A>::value_type {
  return expression::binary::FormTensorProduct(a.expression(), b.expression())
      .coeff();
}

} // namespace zipper::utils::detail

#endif
