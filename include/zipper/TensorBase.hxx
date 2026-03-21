#if !defined(ZIPPER_TENSORBASE_HXX)
#define ZIPPER_TENSORBASE_HXX

#include "TensorBase.hpp"
#include "expression/nullary/MDSpan.hpp"
#include "zipper/detail/declare_operations.hpp"
#include "zipper/expression/binary/ArithmeticExpressions.hpp"
#include "zipper/expression/binary/ZeroAwareOperation.hpp"
#include "zipper/expression/binary/TensorProduct.hpp"
#include "zipper/expression/reductions/Contraction.hpp"
#include "zipper/expression/unary/PartialTrace.hpp"
#include "zipper/expression/unary/ScalarArithmetic.hpp"

namespace zipper {

// Deduction guide from std::mdspan
template <typename T, typename Extents, typename Layout, typename Accessor>
TensorBase(zipper::mdspan<T, Extents, Layout, Accessor>) -> TensorBase<
    expression::nullary::MDSpan<T, Extents, Layout, Accessor>>;

UNARY_DECLARATION(TensorBase, LogicalNot, operator!)
UNARY_DECLARATION(TensorBase, BitNot, operator~)
UNARY_DECLARATION(TensorBase, Negate, operator-)

SCALAR_BINARY_DECLARATION(TensorBase, Plus, operator+)
SCALAR_BINARY_DECLARATION(TensorBase, Minus, operator-)
SCALAR_BINARY_DECLARATION(TensorBase, Multiplies, operator*)
SCALAR_BINARY_DECLARATION(TensorBase, Divides, operator/)

ZERO_AWARE_BINARY_DECLARATION(TensorBase, Plus, operator+)
ZERO_AWARE_BINARY_DECLARATION(TensorBase, Minus, operator-)

// ── tensor_product(a, b) — named wrapper for TensorProduct ──────────
//
// Equivalent to `a * b` for Tensor types, but available as a named
// function for clarity when the multiplication symbol is ambiguous.

template <typename Expr1, typename Expr2>
    requires(concepts::Tensor<std::decay_t<Expr1>> &&
             concepts::Tensor<std::decay_t<Expr2>>)
auto operator*(Expr1&& lhs, Expr2&& rhs) {
  using A = detail::forwarded_expression_t<Expr1>;
  using B = detail::forwarded_expression_t<Expr2>;
  using V = expression::binary::TensorProduct<A, B>;
  return TensorBase<V>(std::in_place,
      std::forward<Expr1>(lhs).expression(),
      std::forward<Expr2>(rhs).expression());
}

template <typename Expr1, typename Expr2>
    requires(concepts::Tensor<std::decay_t<Expr1>> &&
             concepts::Tensor<std::decay_t<Expr2>>)
auto tensor_product(Expr1&& lhs, Expr2&& rhs) {
  return std::forward<Expr1>(lhs) * std::forward<Expr2>(rhs);
}

// ── contract<I, J>(tensor) — partial trace over index pair (I, J) ───
//
// Wraps PartialTrace<Expr, I, J> in a TensorBase.  Contracts
// (sums over) the I-th and J-th indices, reducing rank by 2.
//
// Example: contract<0,1>(T) on a rank-4 tensor yields a rank-2 tensor.

template <rank_type I, rank_type J, typename TensorType>
    requires(concepts::Tensor<std::decay_t<TensorType>>)
auto contract(TensorType&& t) {
  using child_t = detail::forwarded_expression_t<TensorType>;
  using V = expression::unary::PartialTrace<child_t, I, J>;
  return TensorBase<V>(std::in_place,
      std::forward<TensorType>(t).expression());
}

// ── full_contract(tensor) — fold-in-half trace (scalar result) ──────
//
// For a rank-2R tensor, sums expr(i0,...,i_{R-1}, i_{R-1},...,i0) over
// all valid index tuples.  Returns a scalar.  Requires even rank.
//
// Example: full_contract(T) on a rank-4 tensor yields a scalar.

template <typename TensorType>
    requires(concepts::Tensor<std::decay_t<TensorType>> &&
             std::decay_t<TensorType>::extents_type::rank() % 2 == 0)
auto full_contract(TensorType&& t) {
  using child_t = detail::forwarded_expression_t<TensorType>;
  return expression::reductions::Contraction<child_t>(
      std::forward<TensorType>(t).expression())();
}

} // namespace zipper

#endif
