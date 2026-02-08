#if !defined(ZIPPER_EXPRESSION_UNARY_DETAIL_SCALAR_BINARY_OPERATION_DECLARATION_HPP)

#define ZIPPER_EXPRESSION_UNARY_DETAIL_SCALAR_BINARY_OPERATION_DECLARATION_HPP

#include "zipper/concepts/Expression.hpp"

namespace zipper::expression::unary::detail {
template <template <typename, zipper::concepts::QualifiedExpression, bool> class Op,
          template <typename> class BaseType, zipper::concepts::QualifiedExpression Expr,
          typename Scalar>
auto operation_implementation(Expr& lhs, const Scalar& rhs) {
    using OpType = Op<std::decay_t<Scalar>, Expr, true>;
    return BaseType<OpType>(OpType(lhs, rhs, {}));
}
template <template <typename, zipper::concepts::QualifiedExpression, bool> class Op,
          template <typename> class BaseType, zipper::concepts::QualifiedExpression Expr,
          typename Scalar>
auto operation_implementation(const Scalar& lhs, Expr& rhs) {
    using OpType = Op<std::decay_t<Scalar>, Expr, false>;
    return BaseType<OpType>(OpType(lhs, rhs, {}));
}

template <template <zipper::concepts::QualifiedExpression> class Op,
          template <typename> class BaseType, zipper::concepts::QualifiedExpression B>
auto operation_implementation(B& expr) {
    using OpType = Op<B>;
    return BaseType<OpType>(OpType(expr));
}

}  // namespace zipper::expression::unary::detail
#endif
