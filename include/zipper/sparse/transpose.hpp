/// @file transpose.hpp
/// @brief Sparse matrix transpose producing a new sparse matrix.
///
/// Given a sparse matrix A (m x n), computes A^T (n x m) returning a new
/// sparse matrix of the same format (COO or CSR).
///
/// For COO matrices, this simply swaps row and column indices and
/// re-compresses.  For CSR matrices, the operation goes through COO as an
/// intermediary.
///
/// Three interfaces:
///   - `sparse_transpose(COOMatrix)` -> `COOMatrix`
///   - `sparse_transpose(CSRMatrix)` -> `CSRMatrix`
///   - `sparse_transpose_to_coo(CSRMatrix)` -> `COOMatrix`

#if !defined(ZIPPER_SPARSE_TRANSPOSE_HPP)
#define ZIPPER_SPARSE_TRANSPOSE_HPP

#include <zipper/COOMatrix.hpp>
#include <zipper/CSRMatrix.hpp>

namespace zipper::sparse {

/// @brief Transpose a COO matrix, returning a new COO matrix.
///
/// Swaps row and column indices for every stored entry and re-compresses.
///
/// @param A  An m x n COO sparse matrix.
/// @return   An n x m COO sparse matrix equal to A^T.
///
/// @deprecated Use `coo_matrix.transpose().eval()` instead.
template <typename T, index_type Rows, index_type Cols>
[[deprecated("Use .transpose().eval() instead of sparse_transpose()")]]
auto sparse_transpose(const COOMatrix<T, Rows, Cols> &A)
    -> COOMatrix<T, Cols, Rows> {
  COOMatrix<T, Cols, Rows> At;
  if constexpr (!COOMatrix<T, Cols, Rows>::is_static) {
    At = COOMatrix<T, Cols, Rows>(A.extent(1), A.extent(0));
  }

  for (const auto &entry : A) {
    auto [row, col] = entry.multiindex();
    At.emplace(col, row) = entry.value();
  }
  At.compress();
  return At;
}

/// @brief Transpose a CSR matrix, returning a COO matrix.
///
/// @param A  An m x n CSR sparse matrix.
/// @return   An n x m COO sparse matrix equal to A^T.
///
/// @deprecated Use `csr_matrix.transpose().eval()` instead.
template <typename T, index_type Rows, index_type Cols>
[[deprecated("Use .transpose().eval() instead of sparse_transpose_to_coo()")]]
auto sparse_transpose_to_coo(const CSRMatrix<T, Rows, Cols> &A)
    -> COOMatrix<T, Cols, Rows> {
  COOMatrix<T, Cols, Rows> At;
  if constexpr (!COOMatrix<T, Cols, Rows>::is_static) {
    At = COOMatrix<T, Cols, Rows>(A.extent(1), A.extent(0));
  }

  // Iterate over CSR structure directly for efficiency
  const auto &cd = A.compressed_data();
  using Base0 =
      storage::detail::SparseCompressedData<std::remove_const_t<T>, 0>;
  const auto &base = static_cast<const Base0 &>(cd);
  for (const auto &[row, start, sz] : cd.m_spans) {
    for (size_t i = start; i < start + sz; ++i) {
      At.emplace(base.m_data[i].first, row) = base.m_data[i].second;
    }
  }
  At.compress();
  return At;
}

/// @brief Transpose a CSR matrix, returning a new CSR matrix.
///
/// @param A  An m x n CSR sparse matrix.
/// @return   An n x m CSR sparse matrix equal to A^T.
///
/// @deprecated Use `csr_matrix.transpose().eval()` instead.
template <typename T, index_type Rows, index_type Cols>
[[deprecated("Use .transpose().eval() instead of sparse_transpose()")]]
auto sparse_transpose(const CSRMatrix<T, Rows, Cols> &A)
    -> CSRMatrix<T, Cols, Rows> {
  // Inline the transpose-to-coo logic to avoid triggering our own
  // deprecation warning on sparse_transpose_to_coo.
  COOMatrix<T, Cols, Rows> At;
  if constexpr (!COOMatrix<T, Cols, Rows>::is_static) {
    At = COOMatrix<T, Cols, Rows>(A.extent(1), A.extent(0));
  }
  const auto &cd = A.compressed_data();
  using Base0 =
      storage::detail::SparseCompressedData<std::remove_const_t<T>, 0>;
  const auto &base = static_cast<const Base0 &>(cd);
  for (const auto &[row, start, sz] : cd.m_spans) {
    for (size_t i = start; i < start + sz; ++i) {
      At.emplace(base.m_data[i].first, row) = base.m_data[i].second;
    }
  }
  At.compress();
  return CSRMatrix<T, Cols, Rows>(At);
}

} // namespace zipper::sparse

#endif
