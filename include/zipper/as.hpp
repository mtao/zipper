#if !defined(ZIPPER_AS_HPP)
#define ZIPPER_AS_HPP
#include "concepts/Zipper.hpp"
#include "expression/concepts/capabilities.hpp"
#include "expression/unary/Slice.hpp"
#include <utility> // std::in_place

namespace zipper {

// Forward declarations of Base types used by the as_* functions.
template <concepts::Expression T> class ArrayBase;
template <concepts::Expression T> class ContainerBase;
template <concepts::Expression T>
  requires(concepts::QualifiedRankedExpression<T, 1>)
class VectorBase;
template <concepts::Expression T>
  requires(concepts::QualifiedRankedExpression<T, 2>)
class MatrixBase;
template <concepts::Expression T> class FormBase;
template <concepts::Expression T> class TensorBase;

// ---- Squeeze helpers --------------------------------------------------------
//
// When converting between semantic wrapper types (e.g. as_vector on a Matrix),
// dimensions whose static extent is exactly 1 can be implicitly collapsed by
// slicing at index 0.  This lets a Matrix<double,3,1> be treated as a
// Vector<double,3> without the caller having to manually slice.
//
// Only *statically* degenerate dimensions (static_extent == 1) are squeezed.
// Dynamic dimensions are never collapsed, since their extent is unknown at
// compile time.

namespace detail {

/// Count the number of non-degenerate dimensions in an extents type.
/// A dimension is degenerate when its static extent is exactly 1.
/// Dimensions with dynamic_extent are treated as non-degenerate because
/// their actual extent is not known at compile time.
template <typename Extents>
consteval rank_type squeeze_rank() {
  rank_type count = 0;
  for (rank_type d = 0; d < Extents::rank(); ++d) {
    if (Extents::static_extent(d) != 1) ++count;
  }
  return count;
}

/// Per-dimension slice specifier for squeezing.
///
/// Produces static_index_t<0> for dimensions with static extent 1 (collapsing
/// them out of the result), or full_extent_t for all other dimensions
/// (preserving them).
template <typename Extents, rank_type D>
using squeeze_specifier_t = std::conditional_t<Extents::static_extent(D) == 1,
                                               static_index_t<0>,
                                               full_extent_t>;

} // namespace detail

// ---- Unconstrained conversions (any rank) -----------------------------------
//
// These conversions impose no rank constraint.  The target Base type itself
// may still reject unsuitable expressions.

#define ZIPPER_AS_IMPL_(NAME_LOWER, NAME_UPPER)                                \
  template <concepts::Zipper ZipperDerived>                                    \
  auto as_##NAME_LOWER(ZipperDerived &v) {                                     \
    using Expr = typename ZipperDerived::expression_type;                      \
    constexpr static bool make_const =                                         \
        !expression::concepts::WritableExpression<Expr> ||                     \
        std::is_const_v<ZipperDerived>;                                        \
    using ExprC = std::conditional_t<make_const, const Expr, Expr>;            \
    return NAME_UPPER##Base<ExprC &>(std::in_place, v.expression());           \
  }                                                                            \
  template <concepts::Zipper ZipperDerived>                                    \
  auto as_##NAME_LOWER(const ZipperDerived &v) {                               \
    using Expr = typename ZipperDerived::expression_type;                      \
    return NAME_UPPER##Base<const Expr &>(std::in_place, v.expression());      \
  }

ZIPPER_AS_IMPL_(array, Array)
ZIPPER_AS_IMPL_(container, Container)
ZIPPER_AS_IMPL_(form, Form)
ZIPPER_AS_IMPL_(tensor, Tensor)

#undef ZIPPER_AS_IMPL_

// ---- Rank-constrained conversions with implicit squeeze ---------------------
//
// as_vector: target rank 1.  Accepts expressions that already have rank 1,
// or higher-rank expressions whose extra dimensions are all statically
// degenerate (static extent == 1), in which case those dimensions are
// transparently sliced away at index 0.
//
// as_matrix: target rank 2.  Same squeeze logic — all but two dimensions
// must have static extent 1.

/// @brief Convert a Zipper wrapper to VectorBase (rank 1).
///
/// If the source expression has rank 1, wraps it directly.  If the rank is
/// higher but every extra dimension has static extent 1, implicitly squeezes
/// them by slicing at index 0, producing a rank-1 view.
///
/// Writability is preserved: if the source expression is writable and the
/// Zipper reference is non-const, the resulting VectorBase is also writable.
template <concepts::Zipper ZipperDerived>
  requires(
      ZipperDerived::expression_traits::extents_type::rank() == 1 ||
      detail::squeeze_rank<
          typename ZipperDerived::expression_traits::extents_type>() == 1)
auto as_vector(ZipperDerived &v) {
  using Expr = typename ZipperDerived::expression_type;
  constexpr static bool make_const =
      !ZipperDerived::expression_traits::is_writable ||
      std::is_const_v<ZipperDerived>;
  using ExprC = std::conditional_t<make_const, const Expr, Expr>;
  using extents_type = typename ZipperDerived::expression_traits::extents_type;

  if constexpr (extents_type::rank() == 1) {
    return VectorBase<ExprC &>(std::in_place, v.expression());
  } else {
    return [&]<rank_type... Ds>(std::integer_sequence<rank_type, Ds...>) {
      using SliceT = expression::unary::Slice<
          ExprC &, detail::squeeze_specifier_t<extents_type, Ds>...>;
      return VectorBase<SliceT>(
          std::in_place, v.expression(),
          detail::squeeze_specifier_t<extents_type, Ds>{}...);
    }(std::make_integer_sequence<rank_type, extents_type::rank()>{});
  }
}

/// @brief Const overload of as_vector.
template <concepts::Zipper ZipperDerived>
  requires(
      ZipperDerived::expression_traits::extents_type::rank() == 1 ||
      detail::squeeze_rank<
          typename ZipperDerived::expression_traits::extents_type>() == 1)
auto as_vector(const ZipperDerived &v) {
  using Expr = typename ZipperDerived::expression_type;
  using extents_type = typename ZipperDerived::expression_traits::extents_type;

  if constexpr (extents_type::rank() == 1) {
    return VectorBase<const Expr &>(std::in_place, v.expression());
  } else {
    return [&]<rank_type... Ds>(std::integer_sequence<rank_type, Ds...>) {
      using SliceT = expression::unary::Slice<
          const Expr &, detail::squeeze_specifier_t<extents_type, Ds>...>;
      return VectorBase<SliceT>(
          std::in_place, v.expression(),
          detail::squeeze_specifier_t<extents_type, Ds>{}...);
    }(std::make_integer_sequence<rank_type, extents_type::rank()>{});
  }
}

/// @brief Convert a Zipper wrapper to MatrixBase (rank 2).
///
/// If the source expression has rank 2, wraps it directly.  If the rank is
/// higher but all extra dimensions have static extent 1, implicitly squeezes
/// them by slicing at index 0, producing a rank-2 view.
///
/// Writability is preserved: if the source expression is writable and the
/// Zipper reference is non-const, the resulting MatrixBase is also writable.
template <concepts::Zipper ZipperDerived>
  requires(
      ZipperDerived::expression_traits::extents_type::rank() == 2 ||
      detail::squeeze_rank<
          typename ZipperDerived::expression_traits::extents_type>() == 2)
auto as_matrix(ZipperDerived &v) {
  using Expr = typename ZipperDerived::expression_type;
  constexpr static bool make_const =
      !ZipperDerived::expression_traits::is_writable ||
      std::is_const_v<ZipperDerived>;
  using ExprC = std::conditional_t<make_const, const Expr, Expr>;
  using extents_type = typename ZipperDerived::expression_traits::extents_type;

  if constexpr (extents_type::rank() == 2) {
    return MatrixBase<ExprC &>(std::in_place, v.expression());
  } else {
    return [&]<rank_type... Ds>(std::integer_sequence<rank_type, Ds...>) {
      using SliceT = expression::unary::Slice<
          ExprC &, detail::squeeze_specifier_t<extents_type, Ds>...>;
      return MatrixBase<SliceT>(
          std::in_place, v.expression(),
          detail::squeeze_specifier_t<extents_type, Ds>{}...);
    }(std::make_integer_sequence<rank_type, extents_type::rank()>{});
  }
}

/// @brief Const overload of as_matrix.
template <concepts::Zipper ZipperDerived>
  requires(
      ZipperDerived::expression_traits::extents_type::rank() == 2 ||
      detail::squeeze_rank<
          typename ZipperDerived::expression_traits::extents_type>() == 2)
auto as_matrix(const ZipperDerived &v) {
  using Expr = typename ZipperDerived::expression_type;
  using extents_type = typename ZipperDerived::expression_traits::extents_type;

  if constexpr (extents_type::rank() == 2) {
    return MatrixBase<const Expr &>(std::in_place, v.expression());
  } else {
    return [&]<rank_type... Ds>(std::integer_sequence<rank_type, Ds...>) {
      using SliceT = expression::unary::Slice<
          const Expr &, detail::squeeze_specifier_t<extents_type, Ds>...>;
      return MatrixBase<SliceT>(
          std::in_place, v.expression(),
          detail::squeeze_specifier_t<extents_type, Ds>{}...);
    }(std::make_integer_sequence<rank_type, extents_type::rank()>{});
  }
}

} // namespace zipper
#endif
