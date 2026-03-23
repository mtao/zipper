/// @file spqr.hpp
/// @brief SuiteSparseQR (SPQR) sparse QR factorization bindings.
///
/// Provides sparse QR factorization for general (possibly rectangular)
/// matrices via SuiteSparseQR.
///
/// Three-tier API (matching zipper's existing solver pattern):
///   - `spqr_factor(A)`            — factorize, returns SpqrResult
///   - `SpqrResult::solve(b)`      — solve least-squares using stored factors
///   - `spqr_solve(A, b)`          — convenience: factor + solve
///
/// For overdetermined systems (m > n), this computes the least-squares
/// solution.  For square systems, this is equivalent to QR-based direct
/// solve.
///
/// The input matrix must be in CSC format (`CSMatrix` with `layout_left`).
/// Convert via `A.as_csc()` if needed.
///
/// @note Only available when `ZIPPER_HAS_SUITESPARSE` is defined.
/// @note SuiteSparseQR is a C++ library — its header is `SuiteSparseQR.hpp`.

#if !defined(ZIPPER_UTILS_SUITESPARSE_SPQR_HPP)
#define ZIPPER_UTILS_SUITESPARSE_SPQR_HPP

#ifdef ZIPPER_HAS_SUITESPARSE

#include <SuiteSparseQR.hpp>

#include <expected>
#include <memory>

#include <zipper/CSMatrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/solver/result.hpp>
#include <zipper/utils/suitesparse/common.hpp>

namespace zipper::utils::suitesparse {

// ═══════════════════════════════════════════════════════════════════════
// SpqrResult — factorization result with solve capability
// ═══════════════════════════════════════════════════════════════════════

/// Result of a sparse QR factorization via SuiteSparseQR.
///
/// Owns the CHOLMOD common struct and the sparse input (needed by SPQR's
/// solve interface).  The factorization is stored internally by SPQR.
///
/// @tparam N  Static row extent of the matrix, or `dynamic_extent`.
template <index_type N = dynamic_extent> struct SpqrResult {
  /// CHOLMOD common struct (SPQR uses CHOLMOD internally).
  std::shared_ptr<CholmodCommon> common;
  /// Owned copy of the CSC matrix (SPQR needs it for solve).
  CholmodSparsePtr A_cholmod;
  /// Matrix dimensions.
  index_type nrow, ncol;

  /// @brief Solve the least-squares problem min ||Ax - b||_2 using QR.
  ///
  /// For square full-rank systems, this returns the exact solution.
  /// For overdetermined systems (nrow > ncol), this returns the
  /// least-squares solution.
  ///
  /// @param b  Right-hand side vector of length nrow.
  /// @return   `std::expected<Vector<double,N>, SolverError>`.
  template <concepts::Vector BDerived>
  auto solve(const BDerived &b) const
      -> std::expected<Vector<double, N>, solver::SolverError> {
    using Result = std::expected<Vector<double, N>, solver::SolverError>;
    auto *cc = common->get();

    // Convert RHS to CHOLMOD dense.
    auto B = make_cholmod_dense_ptr(to_cholmod_dense(b, cc), cc);
    if (!B) {
      return Result{std::unexpected(solver::SolverError{
          .kind = solver::SolverError::Kind::breakdown,
          .message = "SPQR: failed to allocate dense RHS"})};
    }

    // Solve via SuiteSparseQR.
    // SuiteSparseQR_min2norm solves min ||Ax-b|| for underdetermined,
    // or the exact solution for square systems.
    // For overdetermined, we use backslash (A\b).
    cholmod_dense *X_raw = SuiteSparseQR<double>(
        SPQR_ORDERING_DEFAULT, SPQR_DEFAULT_TOL,
        A_cholmod.get(), B.get(), cc);

    if (!X_raw) {
      return Result{std::unexpected(solver::SolverError{
          .kind = solver::SolverError::Kind::breakdown,
          .message = "SPQR: QR solve failed"})};
    }
    auto X = make_cholmod_dense_ptr(X_raw, cc);

    // The result vector has ncol entries (the solution dimension).
    // Use dynamic_extent for the output since ncol may differ from nrow.
    Vector<double, N> x(ncol);
    const auto *xp = static_cast<const double *>(X->x);
    for (index_type i = 0; i < ncol; ++i) {
      x(i) = xp[i];
    }
    return Result{std::move(x)};
  }
};

// ═══════════════════════════════════════════════════════════════════════
// spqr_factor — sparse QR factorization
// ═══════════════════════════════════════════════════════════════════════

/// @brief Prepare a sparse QR factorization of a general matrix in CSC
///        format via SuiteSparseQR.
///
/// @param A  An m x n matrix in CSC layout.
/// @return   `std::expected<SpqrResult<R>, SolverError>`.
///
/// @note SPQR re-factors internally on each solve call (its C++ wrapper
///       `SuiteSparseQR<double>(ordering, tol, A, B, cc)` does analysis +
///       factor + solve in one shot).  This function validates the input
///       and stores it for later solve calls.  For repeated solves with
///       different RHS, the SuiteSparseQR C API with explicit
///       `SuiteSparseQR_factorize` would be more efficient — that is a
///       future enhancement.
template <typename T, index_type R, index_type C>
auto spqr_factor(const CSMatrix<T, R, C, storage::layout_left> &A)
    -> std::expected<SpqrResult<R>, solver::SolverError> {
  static_assert(std::is_same_v<T, double>,
                "SPQR only supports double precision");
  constexpr index_type N = R;
  using Result = std::expected<SpqrResult<N>, solver::SolverError>;

  auto common = std::make_shared<CholmodCommon>();
  auto *cc = common->get();

  const auto &cd = A.compressed_data();
  const index_type nrow = A.rows();
  const index_type ncol = A.cols();
  const size_t nnz = cd.nnz();

  // Allocate a CHOLMOD-owned copy of the sparse matrix.
  // (SPQR may modify internal representations during solve.)
  cholmod_sparse *Ac_raw = cholmod_l_allocate_sparse(
      nrow, ncol, nnz, /*sorted=*/1, /*packed=*/1,
      /*stype=*/0, CHOLMOD_REAL, cc);
  if (!Ac_raw) {
    return Result{std::unexpected(solver::SolverError{
        .kind = solver::SolverError::Kind::breakdown,
        .message = "SPQR: failed to allocate cholmod_sparse copy"})};
  }

  // Copy data into CHOLMOD sparse.
  auto *p = static_cast<SuiteSparse_long *>(Ac_raw->p);
  auto *i = static_cast<SuiteSparse_long *>(Ac_raw->i);
  auto *x = static_cast<double *>(Ac_raw->x);

  for (index_type j = 0; j <= ncol; ++j) {
    p[j] = static_cast<SuiteSparse_long>(cd.m_indptr[j]);
  }
  for (size_t k = 0; k < nnz; ++k) {
    i[k] = static_cast<SuiteSparse_long>(cd.m_indices[k]);
    x[k] = cd.m_values[k];
  }

  return Result{SpqrResult<N>{
      .common = std::move(common),
      .A_cholmod = make_cholmod_sparse_ptr(Ac_raw, cc),
      .nrow = nrow,
      .ncol = ncol,
  }};
}

// ═══════════════════════════════════════════════════════════════════════
// spqr_solve — convenience: factor + solve
// ═══════════════════════════════════════════════════════════════════════

/// @brief Solve min ||Ax - b||_2 via sparse QR (SPQR).
///
/// @param A  An m x n matrix in CSC layout.
/// @param b  Right-hand side vector of length m.
/// @return   `std::expected<Vector<double,N>, SolverError>`.
template <typename T, index_type R, index_type C, concepts::Vector BDerived>
auto spqr_solve(const CSMatrix<T, R, C, storage::layout_left> &A,
                const BDerived &b)
    -> std::expected<Vector<double, R>, solver::SolverError> {
  auto result = spqr_factor(A);
  if (!result) {
    return std::unexpected(std::move(result.error()));
  }
  return result->solve(b);
}

} // namespace zipper::utils::suitesparse

#endif // ZIPPER_HAS_SUITESPARSE
#endif // ZIPPER_UTILS_SUITESPARSE_SPQR_HPP
