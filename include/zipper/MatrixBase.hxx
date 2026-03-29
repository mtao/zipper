#if !defined(ZIPPER_MATRIXBASE_HXX)
#define ZIPPER_MATRIXBASE_HXX

#include "MatrixBase.hpp"
#include "ArrayBase.hxx"
#include "expression/nullary/MDSpan.hpp"
#include "zipper/detail/declare_operations.hpp"
#include "zipper/expression/binary/ArithmeticExpressions.hpp"
#include "zipper/expression/binary/ZeroAwareOperation.hpp"
#include "zipper/expression/binary/MatrixProduct.hpp"
#include "zipper/expression/binary/MatrixVectorProduct.hpp"
#include "zipper/expression/unary/ScalarArithmetic.hpp"

namespace zipper {

// Deduction guide from std::mdspan
template <typename T, typename Extents, typename Layout, typename Accessor>
MatrixBase(zipper::mdspan<T, Extents, Layout, Accessor>) -> MatrixBase<
    expression::nullary::MDSpan<T, Extents, Layout, Accessor>>;

UNARY_DECLARATION(MatrixBase, LogicalNot, operator!)
UNARY_DECLARATION(MatrixBase, BitNot, operator~)
UNARY_DECLARATION(MatrixBase, Negate, operator-)

SCALAR_BINARY_DECLARATION(MatrixBase, Divides, operator/)

ZERO_AWARE_BINARY_DECLARATION(MatrixBase, Plus, operator+)
ZERO_AWARE_BINARY_DECLARATION(MatrixBase, Minus, operator-)
//

template <concepts::Matrix Expr1, concepts::Matrix Expr2>
auto operator==(Expr1 const &lhs, Expr2 const &rhs) {
  return (lhs.as_array() == rhs.as_array()).all();
}
template <concepts::Matrix Expr1, concepts::Matrix Expr2>
auto operator!=(Expr1 const &lhs, Expr2 const &rhs) {
  return (lhs.as_array() != rhs.as_array()).any();
}

template <typename Expr1, typename Expr2>
    requires(concepts::Matrix<std::decay_t<Expr1>> &&
             concepts::Matrix<std::decay_t<Expr2>>)
auto operator*(Expr1&& lhs, Expr2&& rhs) {
  using A = detail::forwarded_expression_t<Expr1>;
  using B = detail::forwarded_expression_t<Expr2>;

  // Block CSC × CSR: no efficient evaluation strategy exists.
  // Use A.as_csr() * B  or  A * B.as_csc()  instead.
  using LhsPref = typename expression::detail::ExpressionTraits<
      std::decay_t<A>>::preferred_layout;
  using RhsPref = typename expression::detail::ExpressionTraits<
      std::decay_t<B>>::preferred_layout;
  static_assert(
      !detail::is_csc_times_csr_v<LhsPref, RhsPref>,
      "CSC × CSR matrix multiply is not supported — neither operand's "
      "layout gives efficient inner-product traversal. "
      "Convert one side: use A.as_csr() * B  or  A * B.as_csc().");

  using V = expression::binary::MatrixProduct<A, B>;
  return MatrixBase<V>(std::in_place,
      std::forward<Expr1>(lhs).expression(),
      std::forward<Expr2>(rhs).expression());
}

SCALAR_BINARY_DECLARATION(MatrixBase, Multiplies, operator*)

template <typename Expr1, typename Expr2>
    requires(concepts::Matrix<std::decay_t<Expr1>> &&
             concepts::Vector<std::decay_t<Expr2>>)
auto operator*(Expr1&& lhs, Expr2&& rhs) {
  using A = detail::forwarded_expression_t<Expr1>;
  using B = detail::forwarded_expression_t<Expr2>;
  using V = expression::binary::MatrixVectorProduct<A, B>;
  return VectorBase<V>(std::in_place,
      std::forward<Expr1>(lhs).expression(),
      std::forward<Expr2>(rhs).expression());
}

} // namespace zipper

#endif
