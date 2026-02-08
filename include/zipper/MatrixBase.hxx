#if !defined(ZIPPER_MATRIXBASE_HXX)
#define ZIPPER_MATRIXBASE_HXX

#include "MatrixBase.hpp"
#include "ArrayBase.hxx"
#include "expression/nullary/MDSpan.hpp"
#include "zipper/detail/declare_operations.hpp"
#include "zipper/expression/binary/ArithmeticExpressions.hpp"
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

BINARY_DECLARATION(MatrixBase, Plus, operator+)
BINARY_DECLARATION(MatrixBase, Minus, operator-)
//

template <concepts::Matrix Expr1, concepts::Matrix Expr2>
auto operator==(Expr1 const &lhs, Expr2 const &rhs) {
  return (lhs.as_array() == rhs.as_array()).all();
}
template <concepts::Matrix Expr1, concepts::Matrix Expr2>
auto operator!=(Expr1 const &lhs, Expr2 const &rhs) {
  return (lhs.as_array() != rhs.as_array()).any();
}

template <concepts::Matrix Expr1, concepts::Matrix Expr2>
auto operator*(Expr1 const &lhs, Expr2 const &rhs) {
  using V =
      expression::binary::MatrixProduct<const typename Expr1::expression_type,
                                         const typename Expr2::expression_type>;
  return MatrixBase<V>(V(lhs.expression(), rhs.expression()));
}

template <concepts::Matrix Expr>
auto operator*(Expr const &lhs, typename Expr::value_type const &rhs) {
  using V = expression::unary::ScalarMultiplies<
      typename Expr::value_type, const typename Expr::expression_type, true>;
  return MatrixBase<V>(V(lhs.expression(), rhs));
}
template <concepts::Matrix Expr>
auto operator*(typename Expr::value_type const &lhs, Expr const &rhs) {
  using V = expression::unary::ScalarMultiplies<
      typename Expr::value_type, const typename Expr::expression_type, false>;
  return MatrixBase<V>(V(lhs, rhs.expression()));
}

template <concepts::Matrix Expr1, concepts::Vector Expr2>
auto operator*(Expr1 const &lhs, Expr2 const &rhs) {
  using V = expression::binary::MatrixVectorProduct<
      const typename Expr1::expression_type, const typename Expr2::expression_type>;

  return VectorBase<V>(V(lhs.expression(), rhs.expression()));
}

} // namespace zipper

#endif
