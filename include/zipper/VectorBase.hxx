#if !defined(ZIPPER_VECTORBASE_HXX)
#define ZIPPER_VECTORBASE_HXX

#include "VectorBase.hpp"
#include "ArrayBase.hxx"
#include "FormBase.hxx"
#include "expression/nullary/MDSpan.hpp"
#include "zipper/detail/declare_operations.hpp"
#include "zipper/expression/binary/ArithmeticExpressions.hpp"
#include "zipper/expression/binary/ZeroAwareOperation.hpp"
#include "zipper/expression/unary/ScalarArithmetic.hpp"

namespace zipper {

// Deduction guide from std::mdspan
template <typename T, typename Extents, typename Layout, typename Accessor>
VectorBase(zipper::mdspan<T, Extents, Layout, Accessor>) -> VectorBase<
    expression::nullary::MDSpan<T, Extents, Layout, Accessor>>;

// Deduction guides from std::span
template <typename T, std::size_t N>
VectorBase(std::span<T, N>) -> VectorBase<
    expression::nullary::MDSpan<T, extents<N>>>;

template <typename T>
VectorBase(std::span<T, std::dynamic_extent>) -> VectorBase<
    expression::nullary::MDSpan<T, extents<dynamic_extent>>>;

// Deduction guides from std::vector (creates a mutable dynamic-extent view)
template <typename T, typename Alloc>
VectorBase(std::vector<T, Alloc> &) -> VectorBase<
    expression::nullary::MDSpan<T, extents<dynamic_extent>>>;

template <typename T, typename Alloc>
VectorBase(const std::vector<T, Alloc> &) -> VectorBase<
    expression::nullary::MDSpan<const T, extents<dynamic_extent>>>;

// Deduction guides from std::array (creates a view with static extent)
template <typename T, std::size_t N>
VectorBase(std::array<T, N> &) -> VectorBase<
    expression::nullary::MDSpan<T, extents<N>>>;

template <typename T, std::size_t N>
VectorBase(const std::array<T, N> &) -> VectorBase<
    expression::nullary::MDSpan<const T, extents<N>>>;

UNARY_DECLARATION(VectorBase, LogicalNot, operator!)
UNARY_DECLARATION(VectorBase, BitNot, operator~)
UNARY_DECLARATION(VectorBase, Negate, operator-)

SCALAR_BINARY_DECLARATION(VectorBase, Divides, operator/)

ZERO_AWARE_BINARY_DECLARATION(VectorBase, Plus, operator+)
ZERO_AWARE_BINARY_DECLARATION(VectorBase, Minus, operator-)

template <concepts::Vector Expr1, concepts::Vector Expr2>
auto operator==(Expr1 const &lhs, Expr2 const &rhs) -> bool {
  return (lhs.as_array() == rhs.as_array()).all();
}

template <concepts::Vector Expr1, concepts::Vector Expr2>
auto operator!=(Expr1 const &lhs, Expr2 const &rhs) -> bool {
  return (lhs.as_array() != rhs.as_array()).any();
}
SCALAR_BINARY_DECLARATION(VectorBase, Multiplies, operator*)

// Free-function forms of dot product and norm.
// These enable ADL-based customization: external types can provide
// their own dot/norm in their namespace to satisfy the
// inner_product_space / normed_space concepts.

template <concepts::Vector V1, concepts::Vector V2>
auto dot(V1 const &a, V2 const &b) {
    return a.dot(b);
}

template <concepts::Vector V>
auto norm(V const &v) {
    return v.norm();
}

} // namespace zipper

#endif
