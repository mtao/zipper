#if !defined(ZIPPER_CSRVECTOR_HPP)
#define ZIPPER_CSRVECTOR_HPP

/// @file CSRVector.hpp
/// @brief Sparse vector in compressed format.
/// @ingroup user_types
///
/// `CSRVector<T, N>` is a user-facing sparse vector that stores non-zeros
/// in compressed format via `SparseCompressedAccessor`.
///
/// Construction:
///   - From a `COOVector`: `CSRVector v(coo);` or `auto v = coo.to_csr();`
///   - From entries: `CSRVector<double, 100> v(entries);`
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
class CSRVector
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

  CSRVector() = default;
  CSRVector(const CSRVector &) = default;
  CSRVector(CSRVector &&) = default;
  auto operator=(const CSRVector &) -> CSRVector & = default;
  auto operator=(CSRVector &&) -> CSRVector & = default;

  // From compressed accessor
  CSRVector(expression_type &&expr) : Base(std::move(expr)) {}
  CSRVector(const expression_type &expr) : Base(expr) {}

  // From COOVector (compresses internally)
  CSRVector(const COOVector<ValueType, N> &coo)
      : Base(expression_type(coo.expression())) {}

  // From COO expression
  CSRVector(const coo_expression_type &coo)
      : Base(expression_type(coo)) {}

  // Construct from range of SparseEntry<T, 1> (via COO intermediary)
  template <std::ranges::input_range R>
    requires std::same_as<std::ranges::range_value_t<R>,
                          SparseEntry<value_type, 1>>
  CSRVector(const R &entries)
    requires(is_static)
      : CSRVector(COOVector<ValueType, N>(entries)) {}

  template <std::ranges::input_range R>
    requires std::same_as<std::ranges::range_value_t<R>,
                          SparseEntry<value_type, 1>>
  CSRVector(const R &entries, const extents_type &e)
    requires(!is_static)
      : CSRVector(COOVector<ValueType, N>(entries, e)) {}

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
template <typename VT, typename E>
CSRVector(storage::SparseCompressedAccessor<VT, E>)
    -> CSRVector<VT, E::static_extent(0)>;

template <typename VT, index_type N>
CSRVector(const COOVector<VT, N> &) -> CSRVector<VT, N>;

namespace concepts::detail {
template <typename T, index_type N>
struct IsVector<zipper::CSRVector<T, N>> : std::true_type {};
template <typename T, index_type N>
struct IsZipperBase<zipper::CSRVector<T, N>> : std::true_type {};
} // namespace concepts::detail

// ── COOVector::to_csr() definition (needs CSRVector to be complete) ────
template <typename ValueType, index_type N>
auto COOVector<ValueType, N>::to_csr() const -> CSRVector<ValueType, N> {
  return CSRVector<ValueType, N>(*this);
}

} // namespace zipper

#endif
