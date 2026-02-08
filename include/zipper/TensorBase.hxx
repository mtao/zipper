#if !defined(ZIPPER_TENSORBASE_HXX)
#define ZIPPER_TENSORBASE_HXX

#include "TensorBase.hpp"
#include "expression/nullary/MDSpan.hpp"
#include "zipper/detail/declare_operations.hpp"
#include "zipper/expression/binary/ArithmeticExpressions.hpp"
#include "zipper/expression/binary/TensorProduct.hpp"
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

BINARY_DECLARATION(TensorBase, Plus, operator+)
BINARY_DECLARATION(TensorBase, Minus, operator-)

template <concepts::Tensor Expr1, concepts::Tensor Expr2>
auto operator*(Expr1 const &lhs, Expr2 const &rhs) {
  using V = expression::binary::TensorProduct<const typename Expr1::expression_type,
                                              const typename Expr2::expression_type>;
  return TensorBase<V>(V(lhs.expression(), rhs.expression()));
}

} // namespace zipper

#endif
