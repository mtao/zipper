/// @file cholmod.hpp
/// @brief CHOLMOD sparse Cholesky factorization bindings.
///
/// Provides sparse Cholesky factorization for symmetric positive definite
/// (SPD) matrices via SuiteSparse CHOLMOD.
///
/// Three-tier API (matching zipper's existing solver pattern):
///   - `cholmod_factor(A)`             — factorize, returns CholmodResult
///   - `CholmodResult::solve(b)`       — solve using stored factors
///   - `cholmod_solve(A, b)`           — convenience: factor + solve
///
/// The input matrix must be in CSC format (`CSMatrix` with `layout_left`).
/// Convert via `A.as_csc()` if needed.
///
/// @note Only available when `ZIPPER_HAS_SUITESPARSE` is defined.

#if !defined(ZIPPER_UTILS_SUITESPARSE_CHOLMOD_HPP)
#define ZIPPER_UTILS_SUITESPARSE_CHOLMOD_HPP

#ifdef ZIPPER_HAS_SUITESPARSE

#include <cholmod.h>

#include <expected>
#include <memory>

#include <zipper/CSMatrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/solver/result.hpp>
#include <zipper/utils/suitesparse/common.hpp>

namespace zipper::utils::suitesparse {

// ═══════════════════════════════════════════════════════════════════════
// RAII wrapper for cholmod_factor
// ═══════════════════════════════════════════════════════════════════════

struct CholmodFactorDeleter {
  cholmod_common *cc;
  void operator()(cholmod_factor *p) const {
    if (p) {
      cholmod_l_free_factor(&p, cc);
    }
  }
};
using CholmodFactorPtr =
    std::unique_ptr<cholmod_factor, CholmodFactorDeleter>;

// ═══════════════════════════════════════════════════════════════════════
// CholmodResult — factorization result with solve capability
// ═══════════════════════════════════════════════════════════════════════

/// Result of a sparse Cholesky factorization via CHOLMOD.
///
/// Owns the CHOLMOD factor and common struct.  Calling `.solve(b)`
/// performs a sparse triangular solve without re-factoring.
///
/// @tparam N  Static dimension of the matrix (rows/cols), or
///            `dynamic_extent` for runtime-sized.
template <index_type N = dynamic_extent> struct CholmodResult {
  /// Scalar type of the decomposition.
  using value_type = double;

  /// CHOLMOD common struct (must outlive the factor).
  std::shared_ptr<CholmodCommon> common;
  /// CHOLMOD factorization handle.
  CholmodFactorPtr factor;
  /// Matrix dimension (number of rows = number of columns).
  index_type dim;

  /// @brief Solve Ax = b using the stored Cholesky factors.
  ///
  /// @param b  Right-hand side vector of length `dim`.
  /// @return   `std::expected<Vector<double,N>, SolverError>` — the
  ///           solution on success, or a breakdown error on failure.
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
          .message = "CHOLMOD: failed to allocate dense RHS"})};
    }

    // Solve.
    auto X = make_cholmod_dense_ptr(
        cholmod_l_solve(CHOLMOD_A, factor.get(), B.get(), cc), cc);
    if (!X) {
      return Result{std::unexpected(solver::SolverError{
          .kind = solver::SolverError::Kind::breakdown,
          .message = "CHOLMOD: solve failed"})};
    }

    return Result{from_cholmod_dense<N>(X.get())};
  }
};

// ═══════════════════════════════════════════════════════════════════════
// cholmod_factor — sparse Cholesky factorization
// ═══════════════════════════════════════════════════════════════════════

/// @brief Compute the sparse Cholesky factorization of a symmetric
///        positive definite matrix in CSC format.
///
/// @param A      An n x n SPD matrix in CSC layout.
/// @param stype  Symmetry type: 1 = upper triangle stored (default),
///               -1 = lower triangle stored.
/// @return       `std::expected<CholmodResult<N>, SolverError>`.
template <typename T, index_type R, index_type C>
auto cholmod_factor(const CSMatrix<T, R, C, storage::layout_left> &A,
                    int stype = 1)
    -> std::expected<CholmodResult<R>, solver::SolverError> {
  constexpr index_type N = R;
  using Result = std::expected<CholmodResult<N>, solver::SolverError>;

  try {

  auto common = std::make_shared<CholmodCommon>();
  auto *cc = common->get();

  // Create non-owning view of A.
  auto cs = to_cholmod_sparse(A, stype);

  // Analyze (symbolic factorization).
  cholmod_factor *L_raw = cholmod_l_analyze(&cs, cc);
  if (!L_raw) {
    return Result{std::unexpected(solver::SolverError{
        .kind = solver::SolverError::Kind::breakdown,
        .message = "CHOLMOD: symbolic analysis failed"})};
  }
  CholmodFactorPtr L(L_raw, CholmodFactorDeleter{cc});

  // Numeric factorization.
  int ok = cholmod_l_factorize(&cs, L.get(), cc);
  if (!ok || cc->status == CHOLMOD_NOT_POSDEF) {
    return Result{std::unexpected(solver::SolverError{
        .kind = solver::SolverError::Kind::breakdown,
        .message = "CHOLMOD: matrix is not positive definite"})};
  }

  return Result{CholmodResult<N>{
      .common = std::move(common),
      .factor = std::move(L),
      .dim = static_cast<index_type>(A.rows()),
  }};

  } catch (const std::overflow_error &e) {
    return Result{std::unexpected(solver::SolverError{
        .kind = solver::SolverError::Kind::breakdown,
        .message = e.what()})};
  }
}

// ═══════════════════════════════════════════════════════════════════════
// cholmod_solve — convenience: factor + solve in one call
// ═══════════════════════════════════════════════════════════════════════

/// @brief Solve Ax = b via sparse Cholesky (CHOLMOD).
///
/// @param A      An n x n SPD matrix in CSC layout.
/// @param b      Right-hand side vector of length n.
/// @param stype  Symmetry type: 1 = upper stored (default).
/// @return       `std::expected<Vector<double,N>, SolverError>`.
template <typename T, index_type R, index_type C, concepts::Vector BDerived>
auto cholmod_solve(const CSMatrix<T, R, C, storage::layout_left> &A,
                   const BDerived &b, int stype = 1)
    -> std::expected<Vector<double, R>, solver::SolverError> {
  auto result = cholmod_factor(A, stype);
  if (!result) {
    return std::unexpected(std::move(result.error()));
  }
  return result->solve(b);
}

} // namespace zipper::utils::suitesparse

#endif // ZIPPER_HAS_SUITESPARSE
#endif // ZIPPER_UTILS_SUITESPARSE_CHOLMOD_HPP
