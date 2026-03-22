#if !defined(ZIPPER_CSRMATRIX_HPP)
#define ZIPPER_CSRMATRIX_HPP

/// @file CSRMatrix.hpp
/// @brief Sparse matrix in CSR (Compressed Sparse Row) format.
/// @ingroup user_types
///
/// `CSRMatrix<T, Rows, Cols>` is a user-facing sparse matrix that stores
/// non-zeros in compressed row format via `SparseCompressedAccessor`.
///
/// Construction:
///   - From a `COOMatrix`: `CSRMatrix A(coo);` or `auto A = coo.to_csr();`
///   - From entries: `CSRMatrix<double, 3, 3> A(entries);`
///   - From a `SparseCompressedAccessor`: move existing data.
///
/// Mutation:
///   - `coeff_ref(row, col)` returns a reference to an *existing* entry.
///     Throws for missing entries — use COO for structural changes.
///
/// Conversion:
///   - `to_coo()` returns a `COOMatrix` with the same data.

#include "SparseEntry.hpp"
#include "COOMatrix.hpp"
#include "concepts/Matrix.hpp"
#include "MatrixBase.hxx"
#include "storage/SparseCompressedAccessor.hpp"

namespace zipper {

template <typename ValueType, index_type Rows, index_type Cols>
class CSRMatrix
    : public MatrixBase<
          storage::SparseCompressedAccessor<ValueType, zipper::extents<Rows, Cols>>> {
public:
  using expression_type =
      storage::SparseCompressedAccessor<ValueType, zipper::extents<Rows, Cols>>;
  using coo_expression_type =
      storage::SparseCoordinateAccessor<ValueType, zipper::extents<Rows, Cols>>;
  using Base = MatrixBase<expression_type>;
  using value_type = typename Base::value_type;
  using extents_type = typename Base::extents_type;
  using extents_traits = detail::ExtentsTraits<extents_type>;
  using Base::expression;
  using Base::extent;
  using Base::extents;
  constexpr static bool is_static = extents_traits::is_static;

  CSRMatrix() = default;
  CSRMatrix(const CSRMatrix &) = default;
  CSRMatrix(CSRMatrix &&) = default;
  auto operator=(const CSRMatrix &) -> CSRMatrix & = default;
  auto operator=(CSRMatrix &&) -> CSRMatrix & = default;

  // From compressed accessor
  CSRMatrix(expression_type &&expr) : Base(std::move(expr)) {}
  CSRMatrix(const expression_type &expr) : Base(expr) {}

  // From COOMatrix (compresses internally)
  CSRMatrix(const COOMatrix<ValueType, Rows, Cols> &coo)
      : Base(expression_type(coo.expression())) {}

  // From COO expression
  CSRMatrix(const coo_expression_type &coo)
      : Base(expression_type(coo)) {}

  // Construct from range of SparseEntry<T, 2> (via COO intermediary)
  template <std::ranges::input_range R>
    requires std::same_as<std::ranges::range_value_t<R>,
                          SparseEntry<value_type, 2>>
  CSRMatrix(const R &entries)
    requires(is_static)
      : CSRMatrix(COOMatrix<ValueType, Rows, Cols>(entries)) {}

  template <std::ranges::input_range R>
    requires std::same_as<std::ranges::range_value_t<R>,
                          SparseEntry<value_type, 2>>
  CSRMatrix(const R &entries, const extents_type &e)
    requires(!is_static)
      : CSRMatrix(COOMatrix<ValueType, Rows, Cols>(entries, e)) {}

  // ── Element access ────────────────────────────────────────────────────
  auto coeff_ref(index_type row, index_type col) -> value_type & {
    return expression().coeff_ref(row, col);
  }

  auto const_coeff_ref(index_type row, index_type col) const
      -> const value_type & {
    return expression().const_coeff_ref(row, col);
  }

  auto compressed_data() const -> const auto & {
    return expression().compressed_data();
  }

  // ── Conversion ────────────────────────────────────────────────────────
  auto to_coo() const -> COOMatrix<ValueType, Rows, Cols> {
    COOMatrix<ValueType, Rows, Cols> result;
    if constexpr (!is_static) {
      result = COOMatrix<ValueType, Rows, Cols>(extents());
    }
    const auto &cd = compressed_data();
    using Base0 =
        storage::detail::SparseCompressedData<std::remove_const_t<ValueType>, 0>;
    const auto &base = static_cast<const Base0 &>(cd);
    for (const auto &[row, start, sz] : cd.m_spans) {
      for (size_t i = start; i < start + sz; ++i) {
        result.emplace(row, base.m_data[i].first) = base.m_data[i].second;
      }
    }
    result.compress();
    return result;
  }
};

// CTAD
template <typename VT, typename E>
CSRMatrix(storage::SparseCompressedAccessor<VT, E>)
    -> CSRMatrix<VT, E::static_extent(0), E::static_extent(1)>;

template <typename VT, index_type R, index_type C>
CSRMatrix(const COOMatrix<VT, R, C> &) -> CSRMatrix<VT, R, C>;

namespace concepts::detail {
template <typename T, index_type R, index_type C>
struct IsMatrix<zipper::CSRMatrix<T, R, C>> : std::true_type {};
template <typename T, index_type R, index_type C>
struct IsZipperBase<zipper::CSRMatrix<T, R, C>> : std::true_type {};
} // namespace concepts::detail

// ── COOMatrix::to_csr() definition (needs CSRMatrix to be complete) ────
template <typename ValueType, index_type Rows, index_type Cols>
auto COOMatrix<ValueType, Rows, Cols>::to_csr() const
    -> CSRMatrix<ValueType, Rows, Cols> {
  return CSRMatrix<ValueType, Rows, Cols>(*this);
}

} // namespace zipper

#endif
