#if !defined(ZIPPER_FORMBASE_HXX)
#define ZIPPER_FORMBASE_HXX

#include "FormBase.hpp"
#include "TensorBase.hxx"
#include "expression/nullary/MDSpan.hpp"
#include "zipper/detail/declare_operations.hpp"
#include "zipper/expression/binary/ArithmeticExpressions.hpp"
#include "zipper/expression/binary/FormTensorProduct.hpp"
#include "zipper/expression/binary/WedgeProduct.hpp"
#include "zipper/expression/unary/ScalarArithmetic.hpp"

namespace zipper {

// Deduction guide from std::mdspan
template <typename T, typename Extents, typename Layout, typename Accessor>
FormBase(zipper::mdspan<T, Extents, Layout, Accessor>) -> FormBase<
    expression::nullary::MDSpan<T, Extents, Layout, Accessor>>;

UNARY_DECLARATION(FormBase, LogicalNot, operator!)
UNARY_DECLARATION(FormBase, BitNot, operator~)
UNARY_DECLARATION(FormBase, Negate, operator-)

SCALAR_BINARY_DECLARATION(FormBase, Plus, operator+)
SCALAR_BINARY_DECLARATION(FormBase, Minus, operator-)
SCALAR_BINARY_DECLARATION(FormBase, Multiplies, operator*)
SCALAR_BINARY_DECLARATION(FormBase, Divides, operator/)

BINARY_DECLARATION(FormBase, Plus, operator+)
BINARY_DECLARATION(FormBase, Minus, operator-)

template <concepts::Form Expr1, concepts::Form Expr2>
auto operator^(Expr1 const &lhs, Expr2 const &rhs) {
  using V = expression::binary::WedgeProduct<const typename Expr1::expression_type,
                                            const typename Expr2::expression_type>;
  return FormBase<V>(V(lhs.expression(), rhs.expression()));
}

template <concepts::Form Expr1, concepts::Tensor Expr2>
auto operator*(Expr1 const &lhs, Expr2 const &rhs) {
  using V = expression::binary::FormTensorProduct<const typename Expr1::expression_type,
                                                  const typename Expr2::expression_type>;

  if constexpr (V::result_is_form) {
    return FormBase<V>(V(lhs.expression(), rhs.expression()));
  } else {
    return TensorBase<V>(V(lhs.expression(), rhs.expression()));
  }
}
template <concepts::Form Expr1, concepts::Vector Expr2>
auto operator*(Expr1 const &lhs, Expr2 const &rhs) {

  using V = expression::binary::FormTensorProduct<const typename Expr1::expression_type,
                                                  const typename Expr2::expression_type>;
  if constexpr (V::result_is_form) {
    return FormBase<V>(V(lhs.expression(), rhs.expression()));
  } else {
    return TensorBase<V>(V(lhs.expression(), rhs.expression()));
  }
}

} // namespace zipper

#endif
