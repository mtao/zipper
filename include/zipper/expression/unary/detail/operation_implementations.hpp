#if !defined(ZIPPER_EXPRESSION_UNARY_DETAIL_SCALAR_BINARY_OPERATION_DECLARATION_HPP)

#define ZIPPER_EXPRESSION_UNARY_DETAIL_SCALAR_BINARY_OPERATION_DECLARATION_HPP

#include "zipper/concepts/Expression.hpp"
#include <utility> // std::in_place

namespace zipper::expression::unary::detail {

// ── Scalar on right: expression OP scalar ─────────────────────────

/// Lvalue expression: child stored by const reference.
template <template <typename, zipper::concepts::QualifiedExpression, bool> class Op,
          template <typename> class BaseType, zipper::concepts::QualifiedExpression Expr,
          typename Scalar>
auto operation_implementation(const Expr& lhs, const Scalar& rhs) {
    using OpType = Op<std::decay_t<Scalar>, const Expr&, true>;
    return BaseType<OpType>(std::in_place, lhs, rhs);
}

/// Rvalue expression: child stored by value (non-const for moveability).
template <template <typename, zipper::concepts::QualifiedExpression, bool> class Op,
          template <typename> class BaseType, zipper::concepts::QualifiedExpression Expr,
          typename Scalar>
    requires(!std::is_lvalue_reference_v<Expr>)
auto operation_implementation(Expr&& lhs, const Scalar& rhs) {
    using OpType = Op<std::decay_t<Scalar>, std::decay_t<Expr>, true>;
    return BaseType<OpType>(std::in_place, std::move(lhs), rhs);
}

// ── Scalar on left: scalar OP expression ──────────────────────────

/// Lvalue expression: child stored by const reference.
template <template <typename, zipper::concepts::QualifiedExpression, bool> class Op,
          template <typename> class BaseType, zipper::concepts::QualifiedExpression Expr,
          typename Scalar>
auto operation_implementation(const Scalar& lhs, const Expr& rhs) {
    using OpType = Op<std::decay_t<Scalar>, const Expr&, false>;
    return BaseType<OpType>(std::in_place, lhs, rhs);
}

/// Rvalue expression: child stored by value (non-const for moveability).
template <template <typename, zipper::concepts::QualifiedExpression, bool> class Op,
          template <typename> class BaseType, zipper::concepts::QualifiedExpression Expr,
          typename Scalar>
    requires(!std::is_lvalue_reference_v<Expr>)
auto operation_implementation(const Scalar& lhs, Expr&& rhs) {
    using OpType = Op<std::decay_t<Scalar>, std::decay_t<Expr>, false>;
    return BaseType<OpType>(std::in_place, lhs, std::move(rhs));
}

// ── Pure unary: OP(expression) ────────────────────────────────────

/// Lvalue expression: child stored by const reference.
template <template <zipper::concepts::QualifiedExpression> class Op,
          template <typename> class BaseType, zipper::concepts::QualifiedExpression B>
auto operation_implementation(const B& expr) {
    using OpType = Op<const B&>;
    return BaseType<OpType>(std::in_place, expr);
}

/// Rvalue expression: child stored by value (non-const for moveability).
template <template <zipper::concepts::QualifiedExpression> class Op,
          template <typename> class BaseType, zipper::concepts::QualifiedExpression B>
    requires(!std::is_lvalue_reference_v<B>)
auto operation_implementation(B&& expr) {
    using OpType = Op<std::decay_t<B>>;
    return BaseType<OpType>(std::in_place, std::move(expr));
}

}  // namespace zipper::expression::unary::detail
#endif
