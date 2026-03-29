#if !defined(ZIPPER_EXPRESSION_UNARY_DIAGONALEMBED_HPP)
#define ZIPPER_EXPRESSION_UNARY_DIAGONALEMBED_HPP

/// @file DiagonalEmbed.hpp
/// @brief Unary expression that presents a vector as a diagonal matrix.
/// @ingroup expressions_unary sparsity
///
/// `DiagonalEmbed<ExpressionType>` wraps a rank-1 (vector) expression and
/// presents it as a rank-2 square matrix where the diagonal entries are the
/// vector elements and off-diagonal entries are zero.
///
/// This is the inverse of `DiagonalExtract`, which extracts the diagonal of a
/// matrix as a vector.
///
/// The expression is read-only — off-diagonal entries are computed zeros
/// and cannot be written to.
///
/// **Zero-aware sparsity:** DiagonalEmbed has `has_index_set = true`.
/// Its `index_set<D>(other_idx)` returns a `SingleIndexRange{other_idx}`,
/// meaning each row/column has exactly one non-zero entry.
///
/// @code
///   Vector<float, 3> v({2, 3, 4});
///   MatrixBase<DiagonalEmbed<const decltype(v.expression())&>> D(
///       std::in_place, v.expression());
///   // D(0,0) == 2, D(1,1) == 3, D(2,2) == 4
///   // D(0,1) == 0, D(1,0) == 0, etc.
/// @endcode

#include "UnaryExpressionBase.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/assert.hpp"
#include "zipper/expression/detail/IndexSet.hpp"

namespace zipper::expression {
namespace unary {

template <zipper::concepts::QualifiedExpression ExpressionType>
class DiagonalEmbed;

} // namespace unary

// ── ExpressionDetail ────────────────────────────────────────────────────────

template <zipper::concepts::QualifiedExpression ExpressionType>
struct detail::ExpressionDetail<unary::DiagonalEmbed<ExpressionType>> {
  using child_traits = detail::ExpressionTraits<std::decay_t<ExpressionType>>;
  using value_type = typename child_traits::value_type;
  using base_extents_type = typename child_traits::extents_type;

  static_assert(base_extents_type::rank() == 1,
                "DiagonalEmbed requires a rank-1 (vector) child expression");

  constexpr static index_type extent_static() {
    return base_extents_type::static_extent(0);
  }
};

// ── ExpressionTraits ────────────────────────────────────────────────────────

template <zipper::concepts::QualifiedExpression ExpressionType>
struct detail::ExpressionTraits<unary::DiagonalEmbed<ExpressionType>>
    : public unary::detail::DefaultUnaryExpressionTraits<
          ExpressionType,
          zipper::detail::AccessFeatures{.is_const = true,
                                          .is_reference = false}> {
  using _Detail =
      detail::ExpressionDetail<unary::DiagonalEmbed<ExpressionType>>;
  using value_type = typename _Detail::value_type;

  // Custom coeff() needed — off-diagonal returns zero.
  constexpr static bool is_coefficient_consistent = true;
  constexpr static bool is_value_based = false;

  // Output is NxN square matrix.
  using extents_type =
      zipper::extents<_Detail::extent_static(), _Detail::extent_static()>;

  // Structural sparsity: only the diagonal is non-zero.
  constexpr static bool has_index_set = true;
  constexpr static bool has_known_zeros = has_index_set;
};

// ── DiagonalEmbed class ────────────────────────────────────────────────────

namespace unary {

template <zipper::concepts::QualifiedExpression ExpressionType>
class DiagonalEmbed
    : public UnaryExpressionBase<DiagonalEmbed<ExpressionType>,
                                 ExpressionType> {
 public:
  using self_type = DiagonalEmbed<ExpressionType>;
  using traits = zipper::expression::detail::ExpressionTraits<self_type>;
  using detail_type =
      zipper::expression::detail::ExpressionDetail<self_type>;
  using extents_type = typename traits::extents_type;
  using value_type = typename traits::value_type;
  using Base = UnaryExpressionBase<self_type, ExpressionType>;
  using Base::expression;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

  DiagonalEmbed(const DiagonalEmbed &o) : DiagonalEmbed(o.expression()) {}
  DiagonalEmbed(DiagonalEmbed &&o)
      : DiagonalEmbed(std::move(o).expression()) {}
  auto operator=(const DiagonalEmbed &) -> DiagonalEmbed & = delete;
  auto operator=(DiagonalEmbed &&) -> DiagonalEmbed & = delete;

  template <typename U>
    requires std::constructible_from<typename Base::storage_type, U &&>
  DiagonalEmbed(U &&b) : Base(std::forward<U>(b)) {}

  /// Recursively deep-copy child so the result owns all data.
  auto make_owned() const {
    auto owned_child = expression().make_owned();
    return DiagonalEmbed<const decltype(owned_child)>(
        std::move(owned_child));
  }

  /// Both extents equal the child's extent(0) — it's a square matrix.
  constexpr auto extent([[maybe_unused]] rank_type i) const -> index_type {
    ZIPPER_ASSERT(i < 2);
    return expression().extent(0);
  }

  constexpr auto extents() const -> extents_type {
    return extents_traits::make_extents_from(*this);
  }

  /// coeff(r, c): returns child(r) when r == c, else 0.
  template <typename... Args>
  auto coeff(Args &&...args) const -> value_type {
    static_assert(sizeof...(Args) == 2,
                  "DiagonalEmbed::coeff requires exactly 2 indices");
    auto tup = std::make_tuple(static_cast<index_type>(args)...);
    index_type r = std::get<0>(tup);
    index_type c = std::get<1>(tup);
    if (r == c) {
      return expression().coeff(r);
    }
    return value_type{0};
  }

  // ── Index set (sparsity) ─────────────────────────────────────────

  template <rank_type D>
    requires(D < 2)
  auto index_set(index_type other_idx) const
      -> zipper::expression::detail::SingleIndexRange {
    return {other_idx};
  }

  auto col_range_for_row(index_type row) const
      -> zipper::expression::detail::SingleIndexRange {
    return index_set<1>(row);
  }

  auto row_range_for_col(index_type col) const
      -> zipper::expression::detail::SingleIndexRange {
    return index_set<0>(col);
  }
};

// Deduction guides
template <zipper::concepts::QualifiedExpression ExpressionType>
DiagonalEmbed(const ExpressionType &v)
    -> DiagonalEmbed<const ExpressionType &>;

} // namespace unary
} // namespace zipper::expression
#endif
