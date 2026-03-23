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
/// SpqrResult also exposes:
///   - `rank()`        — numerical rank estimate
///   - `permutation()` — fill-reducing column permutation E
///
/// Explicit Q/R extraction:
///   - `spqr_qr(A)`               — returns SpqrQRResult with explicit
///                                   sparse Q and R factors plus column
///                                   permutation E, such that A*E = Q*R
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
#include <vector>

#include <zipper/CSMatrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/solver/result.hpp>
#include <zipper/utils/suitesparse/common.hpp>

namespace zipper::utils::suitesparse {

// ═══════════════════════════════════════════════════════════════════════
// RAII wrapper for SuiteSparseQR_factorization
// ═══════════════════════════════════════════════════════════════════════

struct SpqrFactorizationDeleter {
  cholmod_common *cc;
  void operator()(SuiteSparseQR_factorization<double> *p) const {
    if (p) {
      SuiteSparseQR_free(&p, cc);
    }
  }
};
using SpqrFactorizationPtr =
    std::unique_ptr<SuiteSparseQR_factorization<double>,
                    SpqrFactorizationDeleter>;

// ═══════════════════════════════════════════════════════════════════════
// SpqrResult — factorization result with solve + rank + permutation
// ═══════════════════════════════════════════════════════════════════════

/// Result of a sparse QR factorization via SuiteSparseQR.
///
/// Owns the CHOLMOD common struct and the SPQR factorization handle.
/// The factorization is computed once via `SuiteSparseQR_factorize` and
/// reused for multiple solves without re-factoring.
///
/// @tparam N  Static column extent of the matrix, or `dynamic_extent`.
template <index_type N = dynamic_extent> struct SpqrResult {
  /// Scalar type of the decomposition.
  using value_type = double;

  /// CHOLMOD common struct (SPQR uses CHOLMOD internally).
  std::shared_ptr<CholmodCommon> common;
  /// SPQR factorization handle (owns internal Q in Householder form + R).
  SpqrFactorizationPtr factor;
  /// Matrix dimensions.
  index_type nrow, ncol;

  /// @brief Solve the least-squares problem min ||Ax - b||_2 using QR.
  ///
  /// For square full-rank systems, this returns the exact solution.
  /// For overdetermined systems (nrow > ncol), this returns the
  /// least-squares solution.
  ///
  /// Uses the stored factorization: computes Q'*b via
  /// `SuiteSparseQR_qmult`, then solves R*E'*x = Q'*b via
  /// `SuiteSparseQR_solve`.
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

    // Step 1: Y = Q' * B
    auto Y = make_cholmod_dense_ptr(
        SuiteSparseQR_qmult<double>(SPQR_QTX, factor.get(), B.get(), cc),
        cc);
    if (!Y) {
      return Result{std::unexpected(solver::SolverError{
          .kind = solver::SolverError::Kind::breakdown,
          .message = "SPQR: Q'*b multiplication failed"})};
    }

    // Step 2: X = E * (R \ Y)
    auto X = make_cholmod_dense_ptr(
        SuiteSparseQR_solve<double>(SPQR_RETX_EQUALS_B, factor.get(),
                                    Y.get(), cc),
        cc);
    if (!X) {
      return Result{std::unexpected(solver::SolverError{
          .kind = solver::SolverError::Kind::breakdown,
          .message = "SPQR: R\\(Q'*b) solve failed"})};
    }

    return Result{from_cholmod_dense<N>(X.get())};
  }

  /// @brief Numerical rank estimate of the factored matrix.
  ///
  /// Returns the rank computed during factorization, which is the
  /// number of pivot columns with 2-norm exceeding the tolerance.
  [[nodiscard]] auto rank() const -> index_type {
    return static_cast<index_type>(factor->rank);
  }

  /// @brief Fill-reducing column permutation E.
  ///
  /// Returns a copy of the permutation vector such that Q*R = A*E
  /// (equivalently, A(:,E) = Q*R in MATLAB notation).
  ///
  /// Returns an empty vector if the permutation is the identity.
  [[nodiscard]] auto permutation() const -> std::vector<index_type> {
    if (!factor->Q1fill) {
      return {};
    }
    std::vector<index_type> perm(ncol);
    for (index_type j = 0; j < ncol; ++j) {
      perm[j] = static_cast<index_type>(factor->Q1fill[j]);
    }
    return perm;
  }
};

// ═══════════════════════════════════════════════════════════════════════
// spqr_factor — sparse QR factorization
// ═══════════════════════════════════════════════════════════════════════

/// @brief Compute the sparse QR factorization of a general matrix in CSC
///        format via SuiteSparseQR.
///
/// Uses `SuiteSparseQR_factorize` to compute a reusable factorization.
/// The factorization stores Q in Householder form and R in supernodal
/// block form internally.  Use `.solve(b)` to solve systems, `.rank()`
/// for the numerical rank, and `.permutation()` for the fill-reducing
/// column ordering.
///
/// @param A    An m x n matrix in CSC layout.
/// @param tol  Columns with 2-norm <= tol are treated as zero.
///             Default: `SPQR_DEFAULT_TOL` (uses SPQR's internal heuristic).
/// @return     `std::expected<SpqrResult<C>, SolverError>`.
template <typename T, index_type R, index_type C>
auto spqr_factor(const CSMatrix<T, R, C, storage::layout_left> &A,
                 double tol = SPQR_DEFAULT_TOL)
    -> std::expected<SpqrResult<C>, solver::SolverError> {
  static_assert(std::is_same_v<T, double>,
                "SPQR only supports double precision");
  using Result = std::expected<SpqrResult<C>, solver::SolverError>;

  try {

  auto common = std::make_shared<CholmodCommon>();
  auto *cc = common->get();

  // Create non-owning view of A (validates index bounds).
  auto cs = to_cholmod_sparse(A);

  // Factorize.
  auto *qr_raw = SuiteSparseQR_factorize<double>(
      SPQR_ORDERING_DEFAULT, tol, &cs, cc);
  if (!qr_raw) {
    return Result{std::unexpected(solver::SolverError{
        .kind = solver::SolverError::Kind::breakdown,
        .message = "SPQR: factorization failed"})};
  }

  const index_type nrow = A.rows();
  const index_type ncol = A.cols();

  return Result{SpqrResult<C>{
      .common = std::move(common),
      .factor = SpqrFactorizationPtr(qr_raw, SpqrFactorizationDeleter{cc}),
      .nrow = nrow,
      .ncol = ncol,
  }};

  } catch (const std::overflow_error &e) {
    return Result{std::unexpected(solver::SolverError{
        .kind = solver::SolverError::Kind::breakdown,
        .message = e.what()})};
  }
}

// ═══════════════════════════════════════════════════════════════════════
// spqr_solve — convenience: factor + solve
// ═══════════════════════════════════════════════════════════════════════

/// @brief Solve min ||Ax - b||_2 via sparse QR (SPQR).
///
/// @param A  An m x n matrix in CSC layout.
/// @param b  Right-hand side vector of length m.
/// @return   `std::expected<Vector<double,C>, SolverError>`.
template <typename T, index_type R, index_type C, concepts::Vector BDerived>
auto spqr_solve(const CSMatrix<T, R, C, storage::layout_left> &A,
                const BDerived &b)
    -> std::expected<Vector<double, C>, solver::SolverError> {
  auto result = spqr_factor(A);
  if (!result) {
    return std::unexpected(std::move(result.error()));
  }
  return result->solve(b);
}

// ═══════════════════════════════════════════════════════════════════════
// spqr_rank — convenience: numerical rank via sparse QR
// ═══════════════════════════════════════════════════════════════════════

/// @brief Compute the numerical rank of a sparse matrix via SPQR.
///
/// Convenience function that factors the matrix and returns the rank
/// estimate.  For repeated use, prefer `spqr_factor` and call
/// `.rank()` on the result.
///
/// @param A    An m x n matrix in CSC layout.
/// @param tol  Rank detection tolerance (default: SPQR heuristic).
/// @return     Estimated rank of A, or 0 on failure.
template <typename T, index_type R, index_type C>
auto spqr_rank(const CSMatrix<T, R, C, storage::layout_left> &A,
               double tol = SPQR_DEFAULT_TOL)
    -> index_type {
  auto result = spqr_factor(A, tol);
  if (!result) {
    return 0;
  }
  return result->rank();
}

// ═══════════════════════════════════════════════════════════════════════
// SpqrQRResult — explicit Q and R factors from sparse QR
// ═══════════════════════════════════════════════════════════════════════

/// Result of explicit sparse QR decomposition via SuiteSparseQR.
///
/// Holds the Q and R factors as owned sparse CSC matrices and the
/// column permutation E such that A*E = Q*R (equivalently, A(:,E) = Q*R
/// in MATLAB notation).
///
/// Unlike `SpqrResult`, which stores Q in Householder form for efficient
/// solves, this struct stores Q and R as explicit sparse matrices.
/// This is useful when you need to inspect or manipulate the factors
/// directly (e.g., for rank-revealing analysis, computing projections,
/// or forming the pseudoinverse).
///
/// The decomposition is "economy" sized: Q is m x econ and R is econ x n,
/// where econ = min(m, n) or the estimated rank, depending on SPQR's
/// internal decisions.
///
/// @tparam R  Static row extent of the input matrix (or `dynamic_extent`).
/// @tparam C  Static column extent of the input matrix (or `dynamic_extent`).
template <index_type R = dynamic_extent, index_type C = dynamic_extent>
struct SpqrQRResult {
  /// Scalar type of the decomposition.
  using value_type = double;

  /// Orthonormal factor Q (m x econ, CSC format).
  CSMatrix<double, R, dynamic_extent, storage::layout_left> Q;

  /// Upper trapezoidal factor R (econ x n, CSC format).
  CSMatrix<double, dynamic_extent, C, storage::layout_left> R_factor;

  /// Fill-reducing column permutation E such that A*E = Q*R.
  /// Empty if the permutation is the identity.
  std::vector<index_type> col_perm;

  /// Matrix dimensions.
  index_type nrow, ncol;

  /// Estimated numerical rank (number of columns in the economy-size Q).
  index_type rank_estimate;

  /// @brief Numerical rank estimate.
  [[nodiscard]] auto rank() const -> index_type { return rank_estimate; }
};

// ═══════════════════════════════════════════════════════════════════════
// spqr_qr — explicit sparse QR decomposition
// ═══════════════════════════════════════════════════════════════════════

/// @brief Compute the explicit sparse QR decomposition of a general
///        matrix via SuiteSparseQR.
///
/// Uses the "full form" of SuiteSparseQR that directly outputs the Q
/// and R factors as sparse matrices, along with the column permutation E.
///
/// The decomposition satisfies A*E = Q*R, where:
///   - Q is m x econ, orthonormal columns
///   - R is econ x n, upper trapezoidal
///   - E is a column permutation (fill-reducing ordering)
///
/// @param A    An m x n matrix in CSC layout.
/// @param tol  Columns with 2-norm <= tol are treated as zero.
///             Default: `SPQR_DEFAULT_TOL` (uses SPQR's internal heuristic).
/// @return     `std::expected<SpqrQRResult<R, C>, SolverError>`.
template <typename T, index_type R, index_type C>
auto spqr_qr(const CSMatrix<T, R, C, storage::layout_left> &A,
             double tol = SPQR_DEFAULT_TOL)
    -> std::expected<SpqrQRResult<R, C>, solver::SolverError> {
  static_assert(std::is_same_v<T, double>,
                "SPQR only supports double precision");
  using Result = std::expected<SpqrQRResult<R, C>, solver::SolverError>;

  try {

  auto common = std::make_shared<CholmodCommon>();
  auto *cc = common->get();

  // Create non-owning view of A (validates index bounds).
  auto cs = to_cholmod_sparse(A);

  // Call the full-form SuiteSparseQR that outputs explicit Q, R, E.
  cholmod_sparse *Q_raw = nullptr;
  cholmod_sparse *R_raw = nullptr;
  SuiteSparse_long *E_raw = nullptr;

  const SuiteSparse_long econ = static_cast<SuiteSparse_long>(
      std::min(A.rows(), A.cols()));

  SuiteSparse_long rnk = SuiteSparseQR<double>(
      SPQR_ORDERING_DEFAULT, tol, econ, &cs, &Q_raw, &R_raw, &E_raw, cc);

  // Wrap outputs in RAII handles immediately.
  auto Q_ptr = make_cholmod_sparse_ptr(Q_raw, cc);
  auto R_ptr = make_cholmod_sparse_ptr(R_raw, cc);
  // E_raw is allocated with cholmod_l_malloc; free with cholmod_l_free.
  auto E_deleter = [cc, ncol = A.cols()](SuiteSparse_long *p) {
    if (p) {
      cholmod_l_free(ncol, sizeof(SuiteSparse_long), p, cc);
    }
  };
  std::unique_ptr<SuiteSparse_long, decltype(E_deleter)> E_ptr(
      E_raw, E_deleter);

  if (rnk < 0 || !Q_ptr || !R_ptr) {
    return Result{std::unexpected(solver::SolverError{
        .kind = solver::SolverError::Kind::breakdown,
        .message = "SPQR: explicit QR decomposition failed"})};
  }

  // Convert CHOLMOD sparse matrices to zipper CSMatrix.
  auto Q_mat = from_cholmod_sparse<R, dynamic_extent>(Q_ptr.get());
  auto R_mat = from_cholmod_sparse<dynamic_extent, C>(R_ptr.get());

  // Extract permutation vector.
  std::vector<index_type> perm;
  if (E_ptr) {
    const index_type ncol = A.cols();
    perm.resize(ncol);
    for (index_type j = 0; j < ncol; ++j) {
      perm[j] = static_cast<index_type>(E_ptr.get()[j]);
    }
  }

  return Result{SpqrQRResult<R, C>{
      .Q = std::move(Q_mat),
      .R_factor = std::move(R_mat),
      .col_perm = std::move(perm),
      .nrow = A.rows(),
      .ncol = A.cols(),
      .rank_estimate = static_cast<index_type>(rnk),
  }};

  } catch (const std::overflow_error &e) {
    return Result{std::unexpected(solver::SolverError{
        .kind = solver::SolverError::Kind::breakdown,
        .message = e.what()})};
  }
}

} // namespace zipper::utils::suitesparse

#endif // ZIPPER_HAS_SUITESPARSE
#endif // ZIPPER_UTILS_SUITESPARSE_SPQR_HPP
