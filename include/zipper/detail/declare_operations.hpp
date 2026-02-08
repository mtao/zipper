#if !defined(ZIPPER_DETAIL_DECLARE_OPERATIONS_HPP)
#define ZIPPER_DETAIL_DECLARE_OPERATIONS_HPP

#include "zipper/concepts/Expression.hpp"
#include "zipper/concepts/Zipper.hpp"
#include "zipper/expression/binary/detail/operation_implementations.hpp"
#include "zipper/expression/unary/detail/operation_implementations.hpp"

// Helper: checks if ExprType is derived from BASETYPE<expression_type> or
// BASETYPE<const expression_type>.  This is needed because ZipperBase defines
// expression_type = std::decay_t<Expression>, stripping const.  So
// BASETYPE<const Expr> has expression_type = Expr, but is NOT derived from
// BASETYPE<Expr> — the two are sibling template instantiations.  Checking
// both const and non-const branches handles this uniformly.
#define ZIPPER_DERIVED_FROM_BASE(ExprType, BASETYPE) \
    (std::derived_from<ExprType, zipper::BASETYPE<typename ExprType::expression_type>> || \
     std::derived_from<ExprType, zipper::BASETYPE<const typename ExprType::expression_type>>)

#define SCALAR_BINARY_DECLARATION(BASETYPE, NAME, OP)                \
    template <concepts::Zipper ExprType, typename Scalar>  \
        requires(std::is_same_v<std::decay_t<decltype(std::declval<ExprType>())>, ExprType> && \
                 ZIPPER_DERIVED_FROM_BASE(ExprType, BASETYPE) && \
                 !concepts::Zipper<Scalar> &&             \
                 std::is_scalar_v<Scalar>)                           \
    auto OP(const ExprType& a, const Scalar& b) {                    \
        return expression::unary::detail::operation_implementation<  \
            expression::unary::Scalar##NAME, zipper::BASETYPE,       \
            const typename ExprType::expression_type, Scalar>(a.expression(), b); \
    }                                                                \
    template <concepts::Zipper ExprType, typename Scalar>  \
        requires(std::is_same_v<std::decay_t<decltype(std::declval<ExprType>())>, ExprType> && \
                 ZIPPER_DERIVED_FROM_BASE(ExprType, BASETYPE) && \
                 !concepts::Zipper<Scalar> &&             \
                 std::is_scalar_v<Scalar>)                           \
    auto OP(const Scalar& a, const ExprType& b) {                    \
        return expression::unary::detail::operation_implementation<  \
            expression::unary::Scalar##NAME, zipper::BASETYPE,       \
            const typename ExprType::expression_type, Scalar>(a, b.expression()); \
    }

#define UNARY_DECLARATION(BASETYPE, NAME, OP)                       \
    template <concepts::Zipper ExprType>                  \
        requires(ZIPPER_DERIVED_FROM_BASE(ExprType, BASETYPE)) \
    auto OP(const ExprType& a) {                                    \
        return expression::unary::detail::operation_implementation< \
            expression::unary::NAME, zipper::BASETYPE,              \
            const typename ExprType::expression_type>(a.expression()); \
    }

#define BINARY_DECLARATION(BASETYPE, NAME, OP)                             \
    template <concepts::Zipper ExprType,                        \
              concepts::Zipper ExprType2>                       \
        requires(ZIPPER_DERIVED_FROM_BASE(ExprType, BASETYPE) && \
                 ZIPPER_DERIVED_FROM_BASE(ExprType2, BASETYPE)) \
    auto OP(const ExprType& a, const ExprType2& b) {                       \
        return expression::binary::detail::operation_implementation<       \
            expression::binary::NAME, zipper::BASETYPE,                    \
            const typename ExprType::expression_type, const typename ExprType2::expression_type>( \
            a.expression(), b.expression());                               \
    }
#endif
