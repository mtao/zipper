#if !defined(ZIPPER_DETAIL_DECLARE_OPERATIONS_HPP)
#define ZIPPER_DETAIL_DECLARE_OPERATIONS_HPP

#include "zipper/concepts/Expression.hpp"
#include "zipper/concepts/Zipper.hpp"
#include "zipper/expression/binary/detail/operation_implementations.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"
#include "zipper/expression/unary/detail/operation_implementations.hpp"

// Helper: checks if ExprType (after decay) is derived from BASETYPE<T> for
// various qualifications of T.  This is needed because ZipperBase defines
// expression_type = std::decay_t<Expression>, stripping const/ref.  So
// BASETYPE<const Expr> has expression_type = Expr, but is NOT derived from
// BASETYPE<Expr> — the two are sibling template instantiations.
// We check value, const-value, lvalue-ref, and const-lvalue-ref variants
// so that as_array() etc. can return reference-wrapping wrappers.
//
// NOTE: Uses std::decay_t<ExprType> because ExprType may be a reference
// type when deduced through forwarding references.
#define ZIPPER_DERIVED_FROM_BASE(ExprType, BASETYPE) \
    (std::derived_from<std::decay_t<ExprType>, zipper::BASETYPE<typename std::decay_t<ExprType>::expression_type>> || \
     std::derived_from<std::decay_t<ExprType>, zipper::BASETYPE<const typename std::decay_t<ExprType>::expression_type>> || \
     std::derived_from<std::decay_t<ExprType>, zipper::BASETYPE<typename std::decay_t<ExprType>::expression_type &>> || \
     std::derived_from<std::decay_t<ExprType>, zipper::BASETYPE<const typename std::decay_t<ExprType>::expression_type &>>)

namespace zipper::detail {
/// Deduce the child expression storage type from a forwarding reference to a
/// ZipperBase wrapper.  When the wrapper is an lvalue OR the wrapper's own
/// Expression template parameter is already a reference (e.g.
/// FormBase<const Slice&>), we store by const reference.  Only when the
/// wrapper is an rvalue AND its expression is a value type do we store by
/// value (move).
template <typename Wrapper>
using forwarded_expression_t = std::conditional_t<
    std::is_lvalue_reference_v<Wrapper> ||
    std::is_reference_v<typename std::decay_t<Wrapper>::raw_expression_type>,
    const typename std::decay_t<Wrapper>::expression_type&,
    typename std::decay_t<Wrapper>::expression_type>;
}  // namespace zipper::detail

// ── Scalar × Zipper and Zipper × Scalar ──────────────────────────────
//
// Each macro generates two operator overloads (zipper OP scalar, scalar OP
// zipper).  Each overload takes the Zipper argument by forwarding reference
// so that rvalue temporaries have their expression moved into by-value
// storage while lvalues remain stored by const reference.

#define SCALAR_BINARY_DECLARATION(BASETYPE, NAME, OP)                \
    template <typename ExprType, typename Scalar>                    \
        requires(concepts::Zipper<ExprType> &&                       \
                 ZIPPER_DERIVED_FROM_BASE(ExprType, BASETYPE) &&     \
                 !concepts::Zipper<Scalar> &&                        \
                 std::is_scalar_v<Scalar>)                           \
    auto OP(ExprType&& a, const Scalar& b) {                         \
        return expression::unary::detail::operation_implementation<  \
            expression::unary::Scalar##NAME, zipper::BASETYPE>(      \
            std::forward<ExprType>(a).expression(), b);              \
    }                                                                \
    template <typename ExprType, typename Scalar>                    \
        requires(concepts::Zipper<ExprType> &&                       \
                 ZIPPER_DERIVED_FROM_BASE(ExprType, BASETYPE) &&     \
                 !concepts::Zipper<Scalar> &&                        \
                 std::is_scalar_v<Scalar>)                           \
    auto OP(const Scalar& a, ExprType&& b) {                         \
        return expression::unary::detail::operation_implementation<  \
            expression::unary::Scalar##NAME, zipper::BASETYPE>(      \
            a, std::forward<ExprType>(b).expression());              \
    }

// ── Unary: OP(zipper) ────────────────────────────────────────────────

#define UNARY_DECLARATION(BASETYPE, NAME, OP)                       \
    template <typename ExprType>                                    \
        requires(concepts::Zipper<ExprType> &&                      \
                 ZIPPER_DERIVED_FROM_BASE(ExprType, BASETYPE))      \
    auto OP(ExprType&& a) {                                         \
        return expression::unary::detail::operation_implementation< \
            expression::unary::NAME, zipper::BASETYPE>(             \
            std::forward<ExprType>(a).expression());                \
    }

// ── Binary: zipper OP zipper ─────────────────────────────────────────

#define BINARY_DECLARATION(BASETYPE, NAME, OP)                             \
    template <typename ExprType, typename ExprType2>                       \
        requires(concepts::Zipper<ExprType> &&                             \
                 concepts::Zipper<ExprType2> &&                            \
                 ZIPPER_DERIVED_FROM_BASE(ExprType, BASETYPE) &&           \
                 ZIPPER_DERIVED_FROM_BASE(ExprType2, BASETYPE))            \
    auto OP(ExprType&& a, ExprType2&& b) {                                \
        return expression::binary::detail::operation_implementation<       \
            expression::binary::NAME, zipper::BASETYPE>(                   \
            std::forward<ExprType>(a).expression(),                        \
            std::forward<ExprType2>(b).expression());                      \
    }

// ── Zero-aware binary: zipper OP zipper ──────────────────────────────
//
// Like BINARY_DECLARATION, but uses `if constexpr` in the function body
// to select `ZeroAware##NAME` when at least one operand has known zeros,
// falling back to `NAME` otherwise.  This avoids C++20 constraint
// subsumption issues that would arise from having two competing overloads.

#define ZERO_AWARE_BINARY_DECLARATION(BASETYPE, NAME, OP)                  \
    template <typename ExprType, typename ExprType2>                       \
        requires(concepts::Zipper<ExprType> &&                             \
                 concepts::Zipper<ExprType2> &&                            \
                 ZIPPER_DERIVED_FROM_BASE(ExprType, BASETYPE) &&           \
                 ZIPPER_DERIVED_FROM_BASE(ExprType2, BASETYPE))            \
    auto OP(ExprType&& a, ExprType2&& b) {                                \
        using lhs_expr_t = std::decay_t<                                   \
            decltype(std::forward<ExprType>(a).expression())>;             \
        using rhs_expr_t = std::decay_t<                                   \
            decltype(std::forward<ExprType2>(b).expression())>;            \
        if constexpr (                                                     \
            expression::detail::HasKnownZeros<lhs_expr_t> ||               \
            expression::detail::HasKnownZeros<rhs_expr_t>) {              \
            return expression::binary::detail::operation_implementation<   \
                expression::binary::ZeroAware##NAME, zipper::BASETYPE>(   \
                std::forward<ExprType>(a).expression(),                    \
                std::forward<ExprType2>(b).expression());                  \
        } else {                                                           \
            return expression::binary::detail::operation_implementation<   \
                expression::binary::NAME, zipper::BASETYPE>(              \
                std::forward<ExprType>(a).expression(),                    \
                std::forward<ExprType2>(b).expression());                  \
        }                                                                  \
    }

#endif
