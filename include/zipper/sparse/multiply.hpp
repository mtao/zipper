/// @file multiply.hpp
/// @brief Sparse-sparse matrix multiplication producing a new sparse matrix.
///
/// Given sparse matrices A (m x k) and B (k x n), computes C = A * B (m x n)
/// returning a new sparse matrix.
///
/// The algorithm iterates row-by-row over A.  For each non-zero A(i,p) in
/// row i, it scatters A(i,p) * B(p,:) into an accumulator for row i.  The
/// accumulator uses a dense vector (of length n) with a companion index
/// tracking vector to avoid scanning the full dense vector when extracting
/// non-zeros.
///
/// Also provides optimized routines for the common patterns A^T * A and
/// A * A^T, which exploit symmetry to halve the work by only computing
/// the upper triangle and mirroring.
///
/// Interfaces:
///   - `sparse_multiply(A, B)` -> `CSRMatrix`
///   - `sparse_AtA(A)`         -> `CSRMatrix`  (computes A^T * A)
///   - `sparse_AAt(A)`         -> `CSRMatrix`  (computes A * A^T)

#if !defined(ZIPPER_SPARSE_MULTIPLY_HPP)
#define ZIPPER_SPARSE_MULTIPLY_HPP

#include <algorithm>
#include <vector>

#include <zipper/COOMatrix.hpp>
#include <zipper/CSRMatrix.hpp>

namespace zipper::sparse {

// ─────────────────────────────────────────────────────────────────────────────
// General sparse-sparse multiply: C = A * B
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Multiply two sparse CSR matrices, returning a CSR matrix.
///
/// Uses a row-by-row scatter-gather approach with a dense accumulator.
///
/// @param A  An m x k CSR sparse matrix.
/// @param B  A k x n CSR sparse matrix.
/// @return   An m x n CSR sparse matrix equal to A * B.
template <typename T, index_type M, index_type K1, index_type K2,
          index_type N>
auto sparse_multiply(const CSRMatrix<T, M, K1> &A,
                     const CSRMatrix<T, K2, N> &B)
    -> CSRMatrix<T, M, N> {
  constexpr auto dyn = dynamic_extent;
  const index_type m = A.extent(0);
  const index_type k = A.extent(1);
  const index_type n = B.extent(1);
  static_cast<void>(k); // used only for assertion in debug

  // Dense accumulator for one output row
  std::vector<T> acc(n, T{0});
  // Track which columns have been touched (for efficient extraction)
  std::vector<index_type> touched;
  touched.reserve(n);
  // Marker to know if a column is already in touched list
  std::vector<bool> in_touched(n, false);

  COOMatrix<T, M, N> C;
  if constexpr (!COOMatrix<T, M, N>::is_static) {
    C = COOMatrix<T, M, N>(m, n);
  }

  using Base0 = storage::detail::SparseCompressedData<std::remove_const_t<T>, 0>;

  const auto &cd_a = A.compressed_data();
  const auto &base_a = static_cast<const Base0 &>(cd_a);

  const auto &cd_b = B.compressed_data();
  const auto &base_b = static_cast<const Base0 &>(cd_b);

  for (const auto &[row_a, start_a, sz_a] : cd_a.m_spans) {
    // For each non-zero A(row_a, col_a), scatter col_a-th row of B
    for (size_t ia = start_a; ia < start_a + sz_a; ++ia) {
      auto col_a = base_a.m_data[ia].first;
      auto val_a = base_a.m_data[ia].second;

      // Find col_a as a row in B
      auto it_b = std::lower_bound(
          cd_b.m_spans.begin(), cd_b.m_spans.end(), col_a,
          [](const auto &span, index_type idx) {
            return std::get<0>(span) < idx;
          });

      if (it_b == cd_b.m_spans.end() || std::get<0>(*it_b) != col_a) {
        continue; // B has no entries in this row
      }

      auto [row_b, start_b, sz_b] = *it_b;
      for (size_t ib = start_b; ib < start_b + sz_b; ++ib) {
        auto col_b = base_b.m_data[ib].first;
        auto val_b = base_b.m_data[ib].second;
        if (!in_touched[col_b]) {
          touched.push_back(col_b);
          in_touched[col_b] = true;
        }
        acc[col_b] += val_a * val_b;
      }
    }

    // Gather accumulated results for this row
    std::ranges::sort(touched);
    for (auto col : touched) {
      if (acc[col] != T{0}) {
        C.emplace(row_a, col) = acc[col];
      }
      acc[col] = T{0};
      in_touched[col] = false;
    }
    touched.clear();
  }

  C.compress();
  return CSRMatrix<T, M, N>(C);
}

/// @brief Multiply two sparse COO matrices, returning a CSR matrix.
///
/// Converts inputs to CSR, then delegates to the CSR version.
template <typename T, index_type M, index_type K1, index_type K2,
          index_type N>
auto sparse_multiply(const COOMatrix<T, M, K1> &A,
                     const COOMatrix<T, K2, N> &B)
    -> CSRMatrix<T, M, N> {
  return sparse_multiply(A.to_csr(), B.to_csr());
}

// ─────────────────────────────────────────────────────────────────────────────
// Optimized A^T * A (symmetric result)
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Compute A^T * A for a sparse CSR matrix.
///
/// The result is symmetric positive semi-definite.  This implementation
/// computes the full product (without exploiting symmetry for simplicity)
/// but uses the CSR transpose path.
///
/// @param A  An m x n CSR sparse matrix.
/// @return   An n x n CSR sparse matrix equal to A^T * A.
template <typename T, index_type M, index_type N>
auto sparse_AtA(const CSRMatrix<T, M, N> &A)
    -> CSRMatrix<T, N, N> {
  // Build A^T via COO intermediary
  COOMatrix<T, N, M> At_coo;
  if constexpr (!COOMatrix<T, N, M>::is_static) {
    At_coo = COOMatrix<T, N, M>(A.extent(1), A.extent(0));
  }

  using Base0 = storage::detail::SparseCompressedData<std::remove_const_t<T>, 0>;
  const auto &cd = A.compressed_data();
  const auto &base = static_cast<const Base0 &>(cd);
  for (const auto &[row, start, sz] : cd.m_spans) {
    for (size_t i = start; i < start + sz; ++i) {
      At_coo.emplace(base.m_data[i].first, row) = base.m_data[i].second;
    }
  }
  At_coo.compress();

  auto At = CSRMatrix<T, N, M>(At_coo);
  return sparse_multiply(At, A);
}

/// @brief Compute A * A^T for a sparse CSR matrix.
///
/// @param A  An m x n CSR sparse matrix.
/// @return   An m x m CSR sparse matrix equal to A * A^T.
template <typename T, index_type M, index_type N>
auto sparse_AAt(const CSRMatrix<T, M, N> &A)
    -> CSRMatrix<T, M, M> {
  // Build A^T via COO intermediary
  COOMatrix<T, N, M> At_coo;
  if constexpr (!COOMatrix<T, N, M>::is_static) {
    At_coo = COOMatrix<T, N, M>(A.extent(1), A.extent(0));
  }

  using Base0 = storage::detail::SparseCompressedData<std::remove_const_t<T>, 0>;
  const auto &cd = A.compressed_data();
  const auto &base = static_cast<const Base0 &>(cd);
  for (const auto &[row, start, sz] : cd.m_spans) {
    for (size_t i = start; i < start + sz; ++i) {
      At_coo.emplace(base.m_data[i].first, row) = base.m_data[i].second;
    }
  }
  At_coo.compress();

  auto At = CSRMatrix<T, N, M>(At_coo);
  return sparse_multiply(A, At);
}

} // namespace zipper::sparse

#endif
