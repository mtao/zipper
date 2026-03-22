#if !defined(ZIPPER_COOVECTOR_HPP)
#define ZIPPER_COOVECTOR_HPP

/// @file COOVector.hpp
/// @brief Sparse vector in COO (Coordinate) format.
/// @ingroup user_types
///
/// `COOVector<T, N>` is a user-facing sparse vector that stores non-zeros
/// in coordinate format via `SparseCoordinateAccessor`.
///
/// Construction:
///   - Default (static): `COOVector<double, 100> v;`
///   - Dynamic: `COOVector<double, dynamic_extent> v(n);`
///   - From entries: `COOVector<double, 100> v(entries);`

#include "SparseEntry.hpp"
#include "concepts/Vector.hpp"
#include "VectorBase.hxx"
#include "storage/SparseCoordinateAccessor.hpp"

namespace zipper {

template <typename T, index_type N>
class CSRVector; // forward

template <typename T, index_type N>
class CSVector; // forward

template <typename ValueType, index_type N>
class COOVector
    : public VectorBase<
          storage::SparseCoordinateAccessor<ValueType, zipper::extents<N>>> {
public:
  using expression_type =
      storage::SparseCoordinateAccessor<ValueType, zipper::extents<N>>;
  using Base = VectorBase<expression_type>;
  using value_type = typename Base::value_type;
  using extents_type = typename Base::extents_type;
  using extents_traits = detail::ExtentsTraits<extents_type>;
  using Base::expression;
  using Base::extent;
  using Base::extents;
  constexpr static bool is_static = extents_traits::is_static;

  COOVector() = default;
  COOVector(const COOVector &) = default;
  COOVector(COOVector &&) = default;
  auto operator=(const COOVector &) -> COOVector & = default;
  auto operator=(COOVector &&) -> COOVector & = default;

  /// Assign from any expression with compatible extents.
  template <concepts::Expression Other>
  auto operator=(const Other &other) -> COOVector &
    requires(zipper::utils::extents::assignable_extents_v<
                 typename Other::extents_type, extents_type>)
  {
    expression().assign(other);
    return *this;
  }

  /// Assign from a Zipper-wrapped expression.
  template <concepts::Zipper Other>
  auto operator=(const Other &other) -> COOVector &
    requires(zipper::utils::extents::assignable_extents_v<
                 typename std::decay_t<Other>::extents_type, extents_type>)
  {
    expression().assign(other.expression());
    return *this;
  }

  /// Construct from any expression with compatible extents.
  template <concepts::Expression Other>
  COOVector(const Other &other)
    requires(zipper::utils::extents::assignable_extents_v<
                 typename Other::extents_type, extents_type>)
  {
    if constexpr (!is_static) {
      static_cast<extents_type&>(expression()) =
          extents_traits::convert_from(other.extents());
    }
    expression().assign(other);
  }

  /// Construct from a Zipper-wrapped expression.
  template <concepts::Zipper Other>
  COOVector(const Other &other)
    requires(!std::same_as<std::decay_t<Other>, COOVector> &&
             zipper::utils::extents::assignable_extents_v<
                 typename std::decay_t<Other>::extents_type, extents_type>)
  {
    if constexpr (!is_static) {
      static_cast<extents_type&>(expression()) =
          extents_traits::convert_from(other.extents());
    }
    expression().assign(other.expression());
  }

  // Dynamic extent
  COOVector(index_type size)
    requires(extents_traits::is_dynamic)
      : Base(extents_type(size)) {}

  // Static extent: verify
  COOVector([[maybe_unused]] index_type size)
    requires(is_static)
      : Base() {
    ZIPPER_ASSERT(size == extent(0));
  }

  // From extents
  COOVector(const extents_type &e)
    requires(!is_static)
      : Base(e) {}

  // Move/copy from expression
  COOVector(expression_type &&expr) : Base(std::move(expr)) {}
  COOVector(const expression_type &expr) : Base(expr) {}

  // Construct from range of SparseEntry<T, 1>
  template <std::ranges::input_range R>
    requires std::same_as<std::ranges::range_value_t<R>,
                          SparseEntry<value_type, 1>>
  COOVector(const R &entries)
    requires(is_static)
  {
    for (const auto &[indices, value] : entries) {
      emplace(indices[0]) = value;
    }
    compress();
  }

  template <std::ranges::input_range R>
    requires std::same_as<std::ranges::range_value_t<R>,
                          SparseEntry<value_type, 1>>
  COOVector(const R &entries, const extents_type &e)
    requires(!is_static)
      : Base(e) {
    for (const auto &[indices, value] : entries) {
      emplace(indices[0]) = value;
    }
    compress();
  }

  // ── Sparse mutation ───────────────────────────────────────────────────
  auto emplace(index_type idx) -> value_type & {
    return expression().emplace(idx);
  }

  void compress() { expression().compress(); }

  [[nodiscard]] auto is_compressed() const -> bool {
    return expression().is_compressed();
  }

  auto coeff_ref(index_type idx) -> value_type & {
    return expression().coeff_ref(idx);
  }

  auto const_coeff_ref(index_type idx) const -> const value_type & {
    return expression().const_coeff_ref(idx);
  }

  // ── Conversion ────────────────────────────────────────────────────────
  auto to_csr() const -> CSRVector<ValueType, N>;

  /// Convert to compressed sparse vector format.
  auto to_cs() const -> CSVector<ValueType, N>;

  // ── Iterator access ───────────────────────────────────────────────────
  auto begin() { return expression().begin(); }
  auto end() { return expression().end(); }
  auto begin() const { return expression().begin(); }
  auto end() const { return expression().end(); }
};

// CTAD
template <typename VT, typename E>
COOVector(storage::SparseCoordinateAccessor<VT, E>)
    -> COOVector<VT, E::static_extent(0)>;

namespace concepts::detail {
template <typename T, index_type N>
struct IsVector<zipper::COOVector<T, N>> : std::true_type {};
template <typename T, index_type N>
struct IsZipperBase<zipper::COOVector<T, N>> : std::true_type {};
} // namespace concepts::detail

} // namespace zipper

#endif
