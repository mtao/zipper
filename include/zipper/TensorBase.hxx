#if !defined(ZIPPER_TENSORBASE_HXX)
#define ZIPPER_TENSORBASE_HXX

#include "TensorBase.hpp"
#include "expression/nullary/MDSpan.hpp"
#include "zipper/detail/declare_operations.hpp"
#include "zipper/expression/binary/ArithmeticExpressions.hpp"
#include "zipper/expression/binary/ZeroAwareOperation.hpp"
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

ZERO_AWARE_BINARY_DECLARATION(TensorBase, Plus, operator+)
ZERO_AWARE_BINARY_DECLARATION(TensorBase, Minus, operator-)

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

} // namespace zipper

#endif
