#if !defined(ZIPPER_EXPRESSION_NULLARY_IDENTITY_HPP)
#define ZIPPER_EXPRESSION_NULLARY_IDENTITY_HPP

/// @file Identity.hpp
/// @brief Nullary expression representing the identity tensor.
/// @ingroup expressions_nullary sparsity
///
/// `Identity<T, Indices...>` is a rank-N expression that returns 1 when all
/// indices are equal and 0 otherwise.  For rank-2, this is the identity matrix
/// (Kronecker delta): I(i, j) = (i == j) ? 1 : 0.
///
/// The identity expression owns no data — it generates coefficients on the fly
/// from the index arguments.  It is commonly used to initialise orthogonal
/// matrices (e.g., in QR decomposition) and as a mathematical identity element
/// in matrix algebra.
///
/// @code
///   // Static 3x3 identity matrix
///   auto I = Identity<double, 3, 3>();
///   // I(0,0) == 1, I(0,1) == 0, I(1,1) == 1, ...
///
///   // Dynamic n x n identity matrix
///   auto I_dyn = Identity<double, dynamic_extent, dynamic_extent>(n, n);
/// @endcode
///
/// **Zero-aware sparsity:**  Identity has `has_index_set = true` in its
/// ExpressionTraits.  Its `index_set<D>(other_idx)` returns a
/// `SingleIndexRange{other_idx}`, meaning each row/column has exactly one
/// non-zero entry.  This enables zero-aware matrix products to skip the
/// entire off-diagonal.
///
/// @see zipper::expression::detail::SingleIndexRange — the range type returned
///      by Identity::index_set (exactly one non-zero per row/column).
/// @see zipper::expression::detail::IndexSet — concept satisfied by all
///      range types in the sparsity protocol.
/// @see zipper::expression::unary::TriangularView — another expression with
///      known structural zeros (triangular region).
/// @see zipper::expression::nullary::Unit — unit vector expression (single
///      non-zero in a rank-1 expression).
/// @see zipper::expression::binary::MatrixProduct — uses index_set for
///      zero-aware dot products when an operand is Identity.

#include <utility>

#include "NullaryExpressionBase.hpp"
#include "zipper/detail/make_integer_range_sequence.hpp"
#include "zipper/detail/merge_integer_sequence.hpp"
#include "zipper/detail/pack_index.hpp"
#include "zipper/expression/detail/IndexSet.hpp"

namespace zipper::expression {
namespace nullary {

namespace detail {

template <std::size_t... N>
constexpr static auto
_indicesAllSame(zipper::concepts::IndexPackTuple auto const &t,
                std::index_sequence<N...>) -> bool {
  return (
      (std::get<N>(t) == std::get<zipper::detail::pack_index<0>(N...)>(t)) &&
      ...);
  //
}

constexpr static auto
indicesAllSame(zipper::concepts::IndexPackTuple auto const &t) -> bool {
  return _indicesAllSame(
      t,
      std::make_index_sequence<std::tuple_size_v<std::decay_t<decltype(t)>>>{});
}
} // namespace detail

template <typename T, index_type... Indices>
class Identity : public ExpressionBase<Identity<T, Indices...>>,
                 public zipper::extents<Indices...> {
public:
  using self_type = Identity<T, Indices...>;
  using nullary_base_type = NullaryExpressionBase<self_type>;

  using traits = zipper::expression::detail::ExpressionTraits<self_type>;
  using extents_type = typename traits::extents_type;
  using extents_traits = typename traits::extents_traits;
  using value_type = typename traits::value_type;

  using extents_type::extent;
  using extents_type::rank;
  auto extents() const -> const extents_type & { return *this; }

  Identity(const Identity &) = default;
  Identity(Identity &&) = default;
  auto operator=(const Identity &) -> Identity & = default;
  auto operator=(Identity &&) -> Identity & = default;
  Identity()
    requires(extents_traits::is_static)
  = default;
  Identity(const extents_type &e) : extents_type(e) {}

  template <zipper::concepts::Index... Args>
  Identity(Args &&...args)
      : Identity(extents_type(std::forward<Args>(args)...)) {}

  /// Returns 1 if every input coefficient is the same
  auto coeff(zipper::concepts::Index auto &&...idxs) const -> value_type {
    if constexpr (sizeof...(idxs) == 1) {
      return ((idxs == 0) && ...);
    } else {
      return detail::indicesAllSame(std::make_tuple(idxs...)) ? 1 : 0;
    }
  }

  /// Identity already owns its data — make_owned() returns a copy.
  auto make_owned() const -> Identity { return *this; }

  // ── Index set queries ─────────────────────────────────────────
  // Identity has exactly one non-zero per "row" along any dimension:
  // coeff(i0, i1, ..., iN) == 1 iff i0 == i1 == ... == iN.
  //
  // For rank-2: index_set<1>(row) = SingleIndexRange{row}
  //             index_set<0>(col) = SingleIndexRange{col}
  //
  // For general rank: index_set<D>(other_idx) = SingleIndexRange{other_idx}
  // (the only index along dimension D that can be non-zero is the one
  // equal to the index shared by all other dimensions).

  /// @brief Returns the index set along dimension @p D.
  ///
  /// Since Identity is 1 only when all indices are equal, for any
  /// dimension D and a given index in the other dimensions, the only
  /// non-zero position is where the D-th index equals the others.
  template <rank_type D>
      requires(D < extents_type::rank())
  auto index_set(index_type other_idx) const
      -> zipper::expression::detail::SingleIndexRange {
      return {other_idx};
  }

  /// @brief Backward-compatible alias for index_set.
  template <rank_type D>
      requires(D < extents_type::rank())
  auto nonzero_range(index_type other_idx) const
      -> zipper::expression::detail::SingleIndexRange {
      return index_set<D>(other_idx);
  }

  /// @brief Returns the non-zero column range for a given row (rank-2).
  auto col_range_for_row(index_type row) const
      -> zipper::expression::detail::SingleIndexRange
      requires(extents_type::rank() == 2)
  {
      return index_set<1>(row);
  }

  /// @brief Returns the non-zero row range for a given column (rank-2).
  auto row_range_for_col(index_type col) const
      -> zipper::expression::detail::SingleIndexRange
      requires(extents_type::rank() == 2)
  {
      return index_set<0>(col);
  }

private:
  // =====================================================
  // TODO: this should be used for sparse?
  template <rank_type R, zipper::concepts::Index... Args>
    requires(R < extents_traits::rank &&
             sizeof...(Args) == extents_traits::rank)
  constexpr auto nonZeros(Args &&...args) const -> std::vector<index_type> {
    if (_indicesAllSame(std::make_tuple(args...),
                        zipper::detail::combine_integer_sequence(
                            std::make_index_sequence<R>{},
                            zipper::detail::make_integer_range_sequence<
                                rank_type, R + 1, extents_traits::rank>()))) {
      if constexpr (R == 0 && extents_traits::rank > 1) {
        return {zipper::detail::pack_index<1>(args...)};
      } else {
        return {zipper::detail::pack_index<0>(args...)};
      }

    } else {
      return {};
    }
  }
};
} // namespace nullary

template <typename T, index_type... Indices>
struct detail::ExpressionTraits<nullary::Identity<T, Indices...>>
    : public BasicExpressionTraits<
          T, zipper::extents<Indices...>,
          expression::detail::AccessFeatures::mutable_value(),
          expression::detail::ShapeFeatures::resizable()> {

  /// Identity has structurally known zero regions (off-diagonal is zero).
  constexpr static bool has_index_set = true;

  /// Backward-compatible alias for has_index_set.
  constexpr static bool has_known_zeros = has_index_set;
};
} // namespace zipper::expression

#endif
