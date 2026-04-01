#if !defined(ZIPPER_EXPRESSION_NULLARY_UNIT_HPP)
#define ZIPPER_EXPRESSION_NULLARY_UNIT_HPP

/// @file Unit.hpp
/// @brief Nullary expression representing a unit (basis) vector.
/// @ingroup expressions_nullary sparsity
///
/// `Unit<T, Extent, IndexType>` is a rank-1 expression that returns 1 at a
/// single index and 0 everywhere else — the standard basis vector e_i.  The
/// non-zero index can be a compile-time constant (`std::integral_constant`) or
/// a runtime value (`index_type`).
///
/// Unit vectors are commonly used for:
///   - Extracting columns of a matrix: `A * e_j` gives column j.
///   - Building the identity matrix column-by-column in `inverse_general`.
///   - Representing sparse right-hand sides in linear solvers.
///
/// Three factory functions provide convenient construction:
///   - `unit_vector<T, size, index>()` — static extent + static index.
///   - `unit_vector<T, size>(index)` — static extent + dynamic index.
///   - `unit_vector<T>(size, index)` — dynamic extent + dynamic index.
///
/// @code
///   // Static 3-vector with e_1 = [0, 1, 0]
///   auto e1 = nullary::unit_vector<double, 3, 1>();
///   // e1(0) == 0, e1(1) == 1, e1(2) == 0
///
///   // Dynamic unit vector
///   auto e_j = nullary::unit_vector<double>(n, j);
/// @endcode
///
/// **Zero-aware sparsity:**  Unit has `has_index_set = true` in its
/// ExpressionTraits.  Its `index_set<0>()` returns a
/// `SingleIndexRange{m_index}`, enabling zero-aware matrix-vector products
/// to skip all but one element.
///
/// @see zipper::expression::detail::SingleIndexRange — the range type returned
///      by Unit::index_set (exactly one non-zero).
/// @see zipper::expression::detail::IndexSet — concept satisfied by all
///      range types in the sparsity protocol.
/// @see zipper::expression::nullary::Identity — the rank-2 analogue (identity
///      matrix with one non-zero per row/column).
/// @see zipper::expression::binary::MatrixVectorProduct — uses index_set
///      for zero-aware dot products when the vector is a Unit.

#include <utility>

#include "zipper/detail/no_unique_address.hpp"
#include "zipper/expression/ExpressionBase.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"
#include "zipper/expression/detail/IndexSet.hpp"

namespace zipper::expression {
namespace nullary {

/// A unit vector expression: returns 1 at a single index and 0 elsewhere.
///
/// @tparam T         Value type
/// @tparam Extent    Static extent (or dynamic_extent)
/// @tparam IndexType Either index_type (dynamic index) or
///                   std::integral_constant<index_type, I> (static index)
template <typename T, index_type Extent, typename IndexType>
class Unit : public ExpressionBase<Unit<T, Extent, IndexType>>,
             public zipper::extents<Extent> {
public:
  using self_type = Unit<T, Extent, IndexType>;
  using traits = expression::detail::ExpressionTraits<self_type>;
  using extents_type = typename traits::extents_type;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
  using value_type = typename traits::value_type;

  using extents_type::extent;
  using extents_type::rank;
  auto extents() const -> const extents_type & { return *this; }

  constexpr static bool dynamic_index =
      std::is_same_v<IndexType, index_type>;
  constexpr static bool dynamic_size = extents_traits::is_dynamic;

  // If the size is dynamic we require a dynamic index
  static_assert(dynamic_index || !dynamic_size);

  Unit(const Unit &) = default;
  Unit(Unit &&) = default;
  auto operator=(const Unit &) -> Unit & = default;
  auto operator=(Unit &&) -> Unit & = default;

  /// Static extent + static index
  Unit()
    requires(extents_traits::is_static && !dynamic_index)
      : m_index() {}

  /// Static extent + dynamic index
  Unit(const index_type index)
    requires(!dynamic_size && dynamic_index)
      : m_index(index) {}

  /// Dynamic extent + index
  Unit(const index_type my_extent, IndexType index)
      : extents_type(my_extent), m_index(index) {}

  template <zipper::concepts::Index Idx>
  auto coeff(const Idx &idx) const -> value_type {
    if (static_cast<index_type>(idx) == static_cast<index_type>(m_index)) {
      return value_type(1);
    } else {
      return value_type(0);
    }
  }

  template <rank_type R>
    requires(R == 0)
  constexpr auto nonZeros(index_type) const -> std::array<index_type, 1> {
    return std::array<index_type, 1>{{static_cast<index_type>(m_index)}};
  }

  // ── Index set queries ─────────────────────────────────────────
  // Unit has exactly one non-zero at m_index.
  // For rank-1: index_set<0>() returns SingleIndexRange{m_index}.
  // The overload takes no arguments (rank-1 has only one dimension,
  // and there's no "other" index to condition on).

  /// @brief Returns the index set along dimension @p D.
  ///
  /// For a unit vector, there is exactly one non-zero element.
  template <rank_type D>
      requires(D == 0)
  auto index_set() const
      -> zipper::expression::detail::SingleIndexRange {
      return {static_cast<index_type>(m_index)};
  }

  /// @brief Backward-compatible wrapper for index_set.
  template <rank_type D>
      requires(D == 0)
  auto nonzero_range() const
      -> zipper::expression::detail::SingleIndexRange {
      return index_set<D>();
  }

  /// @brief Convenience alias: returns the non-zero segment (rank-1).
  auto nonzero_segment() const
      -> zipper::expression::detail::SingleIndexRange {
      return index_set<0>();
  }

  /// Unit already owns its data — make_owned() returns a copy.
  auto make_owned() const -> Unit { return *this; }

private:
  ZIPPER_NO_UNIQUE_ADDRESS IndexType m_index;
};

/// Static extent + static index
template <typename T, index_type size, index_type index>
auto unit_vector() {
  return Unit<T, size, std::integral_constant<index_type, index>>{};
}

/// Dynamic extent + dynamic index
template <typename T>
auto unit_vector(index_type size, index_type index) {
  return Unit<T, std::dynamic_extent, index_type>{size, index};
}

/// Static extent + dynamic index
template <typename T, index_type size>
auto unit_vector(index_type index)
  requires(size != std::dynamic_extent)
{
  return Unit<T, size, index_type>(index);
}

} // namespace nullary

template <typename T, index_type Extent, typename IndexType>
struct detail::ExpressionTraits<nullary::Unit<T, Extent, IndexType>>
    : public BasicExpressionTraits<
          T, zipper::extents<Extent>,
          expression::detail::AccessFeatures::mutable_value(),
          expression::detail::ShapeFeatures::fixed()> {

  constexpr static bool is_value_based = false;

  /// Unit has structurally known zero regions (all but one index is zero).
  constexpr static bool has_index_set = true;

  /// Backward-compatible alias for has_index_set.
  constexpr static bool has_known_zeros = has_index_set;
};

} // namespace zipper::expression
#endif
