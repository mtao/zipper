#if !defined(ZIPPER_VECTORBASE_HXX)
#define ZIPPER_VECTORBASE_HXX

#include "VectorBase.hpp"
#include "ArrayBase.hxx"
#include "expression/nullary/MDSpan.hpp"
#include "zipper/detail/declare_operations.hpp"
#include "zipper/expression/binary/ArithmeticExpressions.hpp"
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

BINARY_DECLARATION(VectorBase, Plus, operator+)
BINARY_DECLARATION(VectorBase, Minus, operator-)

template <concepts::Vector Expr1, concepts::Vector Expr2>
auto operator==(Expr1 const &lhs, Expr2 const &rhs) -> bool {
  return (lhs.as_array() == rhs.as_array()).all();
}

template <concepts::Vector Expr1, concepts::Vector Expr2>
auto operator!=(Expr1 const &lhs, Expr2 const &rhs) -> bool {
  return (lhs.as_array() != rhs.as_array()).any();
}
template <concepts::Vector Expr>
auto operator*(Expr const &lhs, typename Expr::value_type const &rhs) {
  using V =
      expression::unary::ScalarMultiplies<typename Expr::value_type,
                                         const typename Expr::expression_type, true>;
  return VectorBase<V>(V(lhs.expression(), rhs));
}
template <concepts::Vector Expr>
auto operator*(typename Expr::value_type const &lhs, Expr const &rhs) {
  using V =
      expression::unary::ScalarMultiplies<typename Expr::value_type,
                                         const typename Expr::expression_type, false>;
  return VectorBase<V>(V(lhs, rhs.expression()));
}

} // namespace zipper

#endif
