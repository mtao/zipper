#if !defined(ZIPPER_COOMATRIX_HPP)
#define ZIPPER_COOMATRIX_HPP

/// @file COOMatrix.hpp
/// @brief Sparse matrix in COO (Coordinate) format.
/// @ingroup user_types
///
/// `COOMatrix<T, Rows, Cols>` is a user-facing sparse matrix that stores
/// non-zeros in coordinate (COO / triplet) format via
/// `SparseCoordinateAccessor`.
///
/// Construction:
///   - Default (static extents): `COOMatrix<double, 3, 3> A;`
///   - Dynamic extents: `COOMatrix<double, dynamic_extent, dynamic_extent> A(m, n);`
///   - From entries: `COOMatrix<double, 3, 3> A(entries);`
///
/// Mutation:
///   - `emplace(row, col)` inserts a new entry (returns ref to value).
///   - `coeff_ref(row, col)` returns a reference to an existing entry.
///   - `compress()` sorts and deduplicates entries.
///
/// Conversion:
///   - `to_csr()` returns a CSR-layout `CSMatrix` with the same data.

#include "SparseEntry.hpp"
#include "concepts/Matrix.hpp"
#include "detail/extents_check.hpp"
#include "MatrixBase.hxx"
#include "storage/SparseCoordinateAccessor.hpp"

namespace zipper {

template <typename T, index_type Rows, index_type Cols, typename LayoutPolicy>
class CSMatrix; // forward

template <typename ValueType, index_type Rows, index_type Cols>
class COOMatrix
    : public MatrixBase<
          storage::SparseCoordinateAccessor<ValueType, zipper::extents<Rows, Cols>>> {
public:
  using expression_type =
      storage::SparseCoordinateAccessor<ValueType, zipper::extents<Rows, Cols>>;
  using Base = MatrixBase<expression_type>;
  using value_type = typename Base::value_type;
  using extents_type = typename Base::extents_type;
  using extents_traits = detail::ExtentsTraits<extents_type>;
  using Base::expression;
  using Base::extent;
  using Base::extents;
  constexpr static bool is_static = extents_traits::is_static;

private:
  template <typename E>
  static auto make_expr_with_extents_(const E &ext) -> expression_type {
    if constexpr (is_static) {
      return expression_type();
    } else {
      return expression_type(extents_traits::convert_from(ext));
    }
  }

public:

  COOMatrix() = default;
  COOMatrix(const COOMatrix &) = default;
  COOMatrix(COOMatrix &&) = default;
  auto operator=(const COOMatrix &) -> COOMatrix & = default;
  auto operator=(COOMatrix &&) -> COOMatrix & = default;

  /// Assign from any expression with compatible extents.
  template <concepts::Expression Other>
  auto operator=(const Other &other) -> COOMatrix &
    requires(zipper::utils::extents::assignable_extents_v<
                 typename Other::extents_type, extents_type>)
  {
    expression().assign(other);
    return *this;
  }

  /// Assign from a Zipper-wrapped expression.
  template <concepts::Zipper Other>
  auto operator=(const Other &other) -> COOMatrix &
    requires(zipper::utils::extents::assignable_extents_v<
                 typename std::decay_t<Other>::extents_type, extents_type>)
  {
    expression().assign(other.expression());
    return *this;
  }

  /// Construct from any expression with compatible extents.
  template <concepts::Expression Other>
  COOMatrix(const Other &other)
    requires(zipper::utils::extents::assignable_extents_v<
                 typename Other::extents_type, extents_type>)
      : Base(make_expr_with_extents_(other.extents()))
  {
    expression().assign(other);
  }

  /// Construct from a Zipper-wrapped expression.
  template <concepts::Zipper Other>
  COOMatrix(const Other &other)
    requires(!std::same_as<std::decay_t<Other>, COOMatrix> &&
             zipper::utils::extents::assignable_extents_v<
                 typename std::decay_t<Other>::extents_type, extents_type>)
      : Base(make_expr_with_extents_(other.extents()))
  {
    expression().assign(other.expression());
  }

  // Dynamic extents: from rows/cols
  COOMatrix(index_type rows, index_type cols)
    requires(extents_traits::is_dynamic)
      : Base(extents_type(rows, cols)) {}

  // Dynamic extents: from extents object
  COOMatrix(const extents_type &e)
    requires(!extents_traits::is_static)
      : Base(e) {}

  // Static extents: verify dimensions match
  COOMatrix([[maybe_unused]] index_type rows,
            [[maybe_unused]] index_type cols)
    requires(is_static)
      : Base() {
    detail::check_extents<extents_type>(rows, cols);
  }

  // Move from expression
  COOMatrix(expression_type &&expr) : Base(std::move(expr)) {}
  COOMatrix(const expression_type &expr) : Base(expr) {}

  // Construct from range of SparseEntry<T, 2>
  template <std::ranges::input_range R>
    requires std::same_as<std::ranges::range_value_t<R>,
                          SparseEntry<value_type, 2>>
  COOMatrix(const R &entries)
    requires(is_static)
  {
    for (const auto &[indices, value] : entries) {
      emplace(indices[0], indices[1]) = value;
    }
    compress();
  }

  template <std::ranges::input_range R>
    requires std::same_as<std::ranges::range_value_t<R>,
                          SparseEntry<value_type, 2>>
  COOMatrix(const R &entries, const extents_type &e)
    requires(!is_static)
      : Base(e) {
    for (const auto &[indices, value] : entries) {
      emplace(indices[0], indices[1]) = value;
    }
    compress();
  }

  // ── Sparse mutation ───────────────────────────────────────────────────
  auto emplace(index_type row, index_type col) -> value_type & {
    return expression().emplace(row, col);
  }

  void compress() { expression().compress(); }

  [[nodiscard]] auto is_compressed() const -> bool {
    return expression().is_compressed();
  }

  auto coeff_ref(index_type row, index_type col) -> value_type & {
    return expression().coeff_ref(row, col);
  }

  auto const_coeff_ref(index_type row, index_type col) const
      -> const value_type & {
    return expression().const_coeff_ref(row, col);
  }

  // ── Conversion ────────────────────────────────────────────────────────
  /// Convert to CSR (compressed sparse row) format.
  auto to_csr() const
      -> CSMatrix<ValueType, Rows, Cols, storage::layout_right>;

  /// Convert to CSC (compressed sparse column) format.
  auto to_csc() const
      -> CSMatrix<ValueType, Rows, Cols, storage::layout_left>;

  /// Convert to compressed sparse format with the given layout.
  template <typename LayoutPolicy>
  auto to_cs() const -> CSMatrix<ValueType, Rows, Cols, LayoutPolicy>;

  // ── Iterator access ───────────────────────────────────────────────────
  auto begin() { return expression().begin(); }
  auto end() { return expression().end(); }
  auto begin() const { return expression().begin(); }
  auto end() const { return expression().end(); }
};

// CTAD
template <typename VT, typename E>
COOMatrix(storage::SparseCoordinateAccessor<VT, E>)
    -> COOMatrix<VT, E::static_extent(0), E::static_extent(1)>;

namespace concepts::detail {
template <typename T, index_type R, index_type C>
struct IsMatrix<zipper::COOMatrix<T, R, C>> : std::true_type {};
template <typename T, index_type R, index_type C>
struct IsZipperBase<zipper::COOMatrix<T, R, C>> : std::true_type {};
} // namespace concepts::detail

} // namespace zipper

#endif
