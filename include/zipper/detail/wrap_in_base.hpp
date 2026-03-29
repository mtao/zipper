#if !defined(ZIPPER_DETAIL_WRAP_IN_BASE_HPP)
#define ZIPPER_DETAIL_WRAP_IN_BASE_HPP

// ── wrap_in_base ─────────────────────────────────────────────────────────
//
// Centralized rank-based wrapping: given an expression type, select the
// appropriate semantic base class (VectorBase for rank 1, MatrixBase for
// rank 2, TensorBase for rank > 2) and wrap it.
//
// This replaces ad-hoc `if constexpr` dispatch on rank at each call site.
//
// Usage:
//   // Type alias:
//   using Wrapped = wrap_in_base_t<MyExpr>;  // e.g. VectorBase<MyExpr>
//
//   // Wrapping function (constructs via std::in_place):
//   auto result = wrap_in_base<MyExpr>(expr_constructor_args...);
//
//   // With explicit base override (e.g. always FormBase):
//   using Wrapped = wrap_in_base_t<MyExpr, FormBase>;

#include "zipper/concepts/Expression.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"

// Forward declarations with matching constraints — must exactly match
// the declarations in as.hpp / concepts/*.hpp / VectorBase.hpp etc.
namespace zipper {
template <concepts::Expression T>
    requires(concepts::QualifiedRankedExpression<T, 1>)
class VectorBase;
template <concepts::Expression T>
    requires(concepts::QualifiedRankedExpression<T, 2>)
class MatrixBase;
template <concepts::Expression T>
class TensorBase;
} // namespace zipper

namespace zipper::detail {

// ── Primary trait: rank -> base class template ─────────────────────────

/// Maps a compile-time rank to the default semantic base class template.
/// rank 1 -> VectorBase, rank 2 -> MatrixBase, rank > 2 -> TensorBase.
/// rank 0 is intentionally unsupported (scalars don't wrap).
template <rank_type Rank>
struct rank_to_base {
    static_assert(
        Rank > 0,
        "rank 0 expressions (scalars) cannot be wrapped in a base class");
    // rank > 2: TensorBase
    template <typename Expr>
    using type = TensorBase<Expr>;
};

template <>
struct rank_to_base<1> {
    template <typename Expr>
    using type = VectorBase<Expr>;
};

template <>
struct rank_to_base<2> {
    template <typename Expr>
    using type = MatrixBase<Expr>;
};

// ── Type alias ─────────────────────────────────────────────────────────

/// Given an expression type, produce the default-wrapped type.
/// Optional second template parameter overrides the base class template.
///
///   wrap_in_base_t<SomeExpr>                // -> VectorBase<SomeExpr> if rank
///   1 wrap_in_base_t<SomeExpr, FormBase>      // -> FormBase<SomeExpr>
///   (forced)
template <
    typename Expr,
    template <typename> typename BaseOverride =
        rank_to_base<expression::detail::ExpressionTraits<
            std::remove_cvref_t<Expr>>::extents_type::rank()>::template type>
using wrap_in_base_t = BaseOverride<Expr>;

// ── Wrapping function ──────────────────────────────────────────────────

/// Construct an expression in-place, wrapped in the rank-appropriate base.
///
///   auto v = wrap_in_base<SliceExpr>(parent_expr, slice_args...);
///   // equivalent to: VectorBase<SliceExpr>(std::in_place, parent_expr,
///   slice_args...)
///
template <typename Expr,
          template <typename> typename BaseOverride = rank_to_base<
              expression::detail::ExpressionTraits<std::remove_cvref_t<Expr>>::
                  extents_type::rank()>::template type,
          typename... Args>
auto wrap_in_base(Args &&...args) -> wrap_in_base_t<Expr, BaseOverride> {
    return wrap_in_base_t<Expr, BaseOverride>(std::in_place,
                                              std::forward<Args>(args)...);
}

} // namespace zipper::detail
#endif
