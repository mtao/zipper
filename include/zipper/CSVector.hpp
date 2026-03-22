#if !defined(ZIPPER_CSVECTOR_HPP)
#define ZIPPER_CSVECTOR_HPP

/// @file CSVector.hpp
/// @brief Sparse vector in compressed format.
/// @ingroup user_types
///
/// `CSVector<T, N>` is the user-facing sparse vector type. Since vectors
/// are rank-1, there is no layout distinction (no CSR vs CSC) — the
/// single dimension is always the compressed dimension.
///
/// Construction:
///   - From a `COOVector`: `CSVector v(coo);` or `auto v = coo.to_cs();`
///   - From entries: `CSVector<double, 100> v(entries);`
///   - From a `SparseCompressedAccessor`: move existing data.
///
/// Mutation:
///   - `coeff_ref(idx)` returns a reference to an *existing* entry.
///     Throws for missing entries — use COO for structural changes.
///
/// Conversion:
///   - `to_coo()` returns a `COOVector` with the same data.

#include "SparseEntry.hpp"
#include "COOVector.hpp"
#include "concepts/Vector.hpp"
#include "VectorBase.hxx"
#include "storage/SparseCompressedAccessor.hpp"

namespace zipper {

template <typename ValueType, index_type N>
class CSVector
    : public VectorBase<
          storage::SparseCompressedAccessor<ValueType, zipper::extents<N>>> {
public:
  using expression_type =
      storage::SparseCompressedAccessor<ValueType, zipper::extents<N>>;
  using coo_expression_type =
      storage::SparseCoordinateAccessor<ValueType, zipper::extents<N>>;
  using Base = VectorBase<expression_type>;
  using value_type = typename Base::value_type;
  using extents_type = typename Base::extents_type;
  using extents_traits = detail::ExtentsTraits<extents_type>;
  using Base::expression;
  using Base::extent;
  using Base::extents;
  constexpr static bool is_static = extents_traits::is_static;

  CSVector() = default;
  CSVector(const CSVector &) = default;
  CSVector(CSVector &&) = default;
  auto operator=(const CSVector &) -> CSVector & = default;
  auto operator=(CSVector &&) -> CSVector & = default;

  /// Assign from any expression with compatible extents.
  template <concepts::Expression Other>
  auto operator=(const Other &other) -> CSVector &
    requires(zipper::utils::extents::assignable_extents_v<
                 typename Other::extents_type, extents_type>)
  {
    expression().assign(other);
    return *this;
  }

  /// Assign from a Zipper-wrapped expression.
  template <concepts::Zipper Other>
  auto operator=(const Other &other) -> CSVector &
    requires(zipper::utils::extents::assignable_extents_v<
                 typename std::decay_t<Other>::extents_type, extents_type>)
  {
    expression().assign(other.expression());
    return *this;
  }

  /// Construct from any expression with compatible extents.
  template <concepts::Expression Other>
  CSVector(const Other &other)
    requires(zipper::utils::extents::assignable_extents_v<
                 typename Other::extents_type, extents_type>)
  {
    if constexpr (!is_static) {
      static_cast<extents_type &>(expression()) =
          extents_traits::convert_from(other.extents());
    }
    expression().assign(other);
  }

  /// Construct from a Zipper-wrapped expression.
  template <concepts::Zipper Other>
  CSVector(const Other &other)
    requires(!std::same_as<std::decay_t<Other>, CSVector> &&
             zipper::utils::extents::assignable_extents_v<
                 typename std::decay_t<Other>::extents_type, extents_type>)
  {
    if constexpr (!is_static) {
      static_cast<extents_type &>(expression()) =
          extents_traits::convert_from(other.extents());
    }
    expression().assign(other.expression());
  }

  // From compressed accessor (any layout — rank-1 is layout-agnostic)
  template <typename LP>
  CSVector(storage::SparseCompressedAccessor<ValueType, zipper::extents<N>, LP> &&expr)
      : Base(expression_type(std::move(expr.compressed_data()),
                             expr.extents())) {}
  CSVector(expression_type &&expr) : Base(std::move(expr)) {}
  CSVector(const expression_type &expr) : Base(expr) {}

  // From COOVector (compresses internally)
  CSVector(const COOVector<ValueType, N> &coo)
      : Base(expression_type(coo.expression())) {}

  // From COO expression
  CSVector(const coo_expression_type &coo)
      : Base(expression_type(coo)) {}

  // Construct from range of SparseEntry<T, 1> (via COO intermediary)
  template <std::ranges::input_range R>
    requires std::same_as<std::ranges::range_value_t<R>,
                          SparseEntry<value_type, 1>>
  CSVector(const R &entries)
    requires(is_static)
      : CSVector(COOVector<ValueType, N>(entries)) {}

  template <std::ranges::input_range R>
    requires std::same_as<std::ranges::range_value_t<R>,
                          SparseEntry<value_type, 1>>
  CSVector(const R &entries, const extents_type &e)
    requires(!is_static)
      : CSVector(COOVector<ValueType, N>(entries, e)) {}

  // ── Element access ────────────────────────────────────────────────────
  auto coeff_ref(index_type idx) -> value_type & {
    return expression().coeff_ref(idx);
  }

  auto const_coeff_ref(index_type idx) const -> const value_type & {
    return expression().const_coeff_ref(idx);
  }

  auto compressed_data() const -> const auto & {
    return expression().compressed_data();
  }

  // ── Conversion ────────────────────────────────────────────────────────
  auto to_coo() const -> COOVector<ValueType, N> {
    COOVector<ValueType, N> result;
    if constexpr (!is_static) {
      result = COOVector<ValueType, N>(extents());
    }
    const auto &cd = compressed_data();
    for (const auto &[idx, val] : cd.m_data) {
      result.emplace(idx) = val;
    }
    result.compress();
    return result;
  }
};

// CTAD
template <typename VT, typename E, typename LP>
CSVector(storage::SparseCompressedAccessor<VT, E, LP>)
    -> CSVector<VT, E::static_extent(0)>;

template <typename VT, index_type N>
CSVector(const COOVector<VT, N> &) -> CSVector<VT, N>;

namespace concepts::detail {
template <typename T, index_type N>
struct IsVector<zipper::CSVector<T, N>> : std::true_type {};
template <typename T, index_type N>
struct IsZipperBase<zipper::CSVector<T, N>> : std::true_type {};
} // namespace concepts::detail

// ── COOVector::to_cs() definition (needs CSVector to be complete) ──────
template <typename ValueType, index_type N>
auto COOVector<ValueType, N>::to_cs() const -> CSVector<ValueType, N> {
  return CSVector<ValueType, N>(*this);
}

} // namespace zipper

#endif
