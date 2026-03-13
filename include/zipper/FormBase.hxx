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

template <typename Expr1, typename Expr2>
    requires(concepts::Form<std::decay_t<Expr1>> &&
             concepts::Form<std::decay_t<Expr2>>)
auto operator^(Expr1&& lhs, Expr2&& rhs) {
  using A = detail::forwarded_expression_t<Expr1>;
  using B = detail::forwarded_expression_t<Expr2>;
  using V = expression::binary::WedgeProduct<A, B>;
  return FormBase<V>(std::in_place,
      std::forward<Expr1>(lhs).expression(),
      std::forward<Expr2>(rhs).expression());
}

template <typename Expr1, typename Expr2>
    requires(concepts::Form<std::decay_t<Expr1>> &&
             concepts::Tensor<std::decay_t<Expr2>>)
auto operator*(Expr1&& lhs, Expr2&& rhs) {
  using A = detail::forwarded_expression_t<Expr1>;
  using B = detail::forwarded_expression_t<Expr2>;
  using V = expression::binary::FormTensorProduct<A, B>;

  if constexpr (V::result_is_form) {
    return FormBase<V>(std::in_place,
        std::forward<Expr1>(lhs).expression(),
        std::forward<Expr2>(rhs).expression());
  } else {
    return TensorBase<V>(std::in_place,
        std::forward<Expr1>(lhs).expression(),
        std::forward<Expr2>(rhs).expression());
  }
}
template <typename Expr1, typename Expr2>
    requires(concepts::Form<std::decay_t<Expr1>> &&
             concepts::Vector<std::decay_t<Expr2>>)
auto operator*(Expr1&& lhs, Expr2&& rhs) {
  using A = detail::forwarded_expression_t<Expr1>;
  using B = detail::forwarded_expression_t<Expr2>;
  using V = expression::binary::FormTensorProduct<A, B>;
  if constexpr (V::result_is_form) {
    return FormBase<V>(std::in_place,
        std::forward<Expr1>(lhs).expression(),
        std::forward<Expr2>(rhs).expression());
  } else {
    return TensorBase<V>(std::in_place,
        std::forward<Expr1>(lhs).expression(),
        std::forward<Expr2>(rhs).expression());
  }
}

} // namespace zipper

#endif
