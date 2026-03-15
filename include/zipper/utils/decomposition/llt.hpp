/// @file llt.hpp
/// @brief Cholesky (LLT) decomposition for symmetric positive definite
///        matrices.
///
/// Given a symmetric positive definite (SPD) matrix A, the Cholesky
/// decomposition factors it as:
///
///   A = L * L^T
///
/// where L is a lower triangular matrix with strictly positive diagonal
/// entries.
///
/// The algorithm processes columns left to right.  For each column j:
///
///   L(j,j) = sqrt( A(j,j) - sum_{k=0}^{j-1} L(j,k)^2 )
///   L(i,j) = ( A(i,j) - sum_{k=0}^{j-1} L(i,k) * L(j,k) ) / L(j,j)
///            for i > j
///
/// Two interfaces:
///   - `llt(A)`        -- returns the lower triangular factor L.
///   - `llt_solve(A,b)` -- solves Ax = b via Cholesky factorisation.
///
/// The solve proceeds in two stages:
///   1. Forward substitution:  L * y = b
///   2. Back substitution:     L^T * x = y
///
/// Both the decomposition and solve detect non-positive-definite matrices
/// (a non-positive diagonal element during factorisation) and return a
/// `SolverError`.
///
/// Complexity: O(n^3 / 3) for the factorisation, O(n^2) for each solve.

#if !defined(ZIPPER_UTILS_DECOMPOSITION_LLT_HPP)
#define ZIPPER_UTILS_DECOMPOSITION_LLT_HPP

#include <cmath>
#include <expected>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/expression/unary/TriangularView.hpp>
#include <zipper/utils/solver/result.hpp>

namespace zipper::utils::decomposition {

// ─────────────────────────────────────────────────────────────────────────────
// Result type
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a Cholesky (LLT) decomposition.
///
/// L is an n x n lower triangular matrix such that A = L * L^T.
///
/// Calling `.solve(b)` performs forward/back substitution using the stored
/// factor to solve A*x = b without re-factoring.
template <typename T, index_type N> struct LLTResult {
  /// Scalar type of the decomposition.
  using value_type = T;

  /// Lower triangular Cholesky factor L (n x n).
  Matrix<T, N, N> L;

  /// @brief Solve A*x = b using the stored Cholesky factor.
  ///
  /// Performs:
  ///   1. Forward substitution:  L * y = b
  ///   2. Back substitution:     L^T * x = y
  ///
  /// @param b  Right-hand side vector of length n.
  /// @return   `std::expected<Vector<T,N>, SolverError>` — the solution on
  ///           success, or a breakdown error if a zero pivot is encountered.
  template <concepts::Vector BDerived>
  auto solve(const BDerived &b) const
      -> std::expected<Vector<T, N>, solver::SolverError> {
    using ResultVec = Vector<T, N>;
    using Result = std::expected<ResultVec, solver::SolverError>;

    const index_type n = L.extent(0);

    // 1. Forward substitution: L * y = b.
    auto L_lower =
        expression::triangular_view<expression::TriangularMode::Lower>(L);
    auto y_result = L_lower.solve(b);

    if (!y_result) {
      return Result{std::unexpected(std::move(y_result.error()))};
    }

    // 2. Back substitution: L^T * x = y.
    //    Build L^T explicitly since TriangularView masks entries but does
    //    not transpose.
    Matrix<T, N, N> Lt(n, n);
    Lt = expression::nullary::Constant(T{0}, Lt.extents());
    for (index_type i = 0; i < n; ++i) {
      for (index_type j = i; j < n; ++j) {
        Lt(i, j) = L(j, i);
      }
    }

    auto Lt_upper =
        expression::triangular_view<expression::TriangularMode::Upper>(Lt);
    auto x_result = Lt_upper.solve(*y_result);

    if (!x_result) {
      return Result{std::unexpected(std::move(x_result.error()))};
    }

    return Result{std::move(*x_result)};
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Cholesky factorisation
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Cholesky (LLT) decomposition of a symmetric positive definite
///        matrix.
///
/// @param A  An n x n symmetric positive definite matrix.
/// @return   `std::expected<LLTResult<T,N>, SolverError>` — the factor L on
///           success, or a breakdown error if A is not positive definite.
template <concepts::Matrix Derived>
auto llt(const Derived &A)
    -> std::expected<
        LLTResult<typename std::decay_t<Derived>::value_type,
                  std::decay_t<Derived>::extents_type::static_extent(0)>,
        solver::SolverError> {
  using AType = std::decay_t<Derived>;
  using T = typename AType::value_type;
  constexpr index_type N = AType::extents_type::static_extent(0);
  using Result =
      std::expected<LLTResult<T, N>, solver::SolverError>;

  const index_type n = A.extent(0);

  // Initialise L to zero.
  Matrix<T, N, N> L(n, n);
  L = expression::nullary::Constant(T{0}, L.extents());

  for (index_type j = 0; j < n; ++j) {
    // Compute the diagonal element L(j,j).
    T sum = T{0};
    for (index_type k = 0; k < j; ++k) {
      sum += L(j, k) * L(j, k);
    }
    T diag = A(j, j) - sum;

    if (diag <= T{0}) {
      return Result{std::unexpected(solver::SolverError{
          .kind = solver::SolverError::Kind::breakdown,
          .message =
              "LLT decomposition: matrix is not positive definite "
              "(non-positive diagonal at column " +
              std::to_string(j) + ")"})};
    }

    L(j, j) = std::sqrt(diag);

    // Compute the sub-diagonal elements L(i,j) for i > j.
    for (index_type i = j + 1; i < n; ++i) {
      T s = T{0};
      for (index_type k = 0; k < j; ++k) {
        s += L(i, k) * L(j, k);
      }
      L(i, j) = (A(i, j) - s) / L(j, j);
    }
  }

  return Result{LLTResult<T, N>{.L = std::move(L)}};
}

// ─────────────────────────────────────────────────────────────────────────────
// Cholesky solve
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Solve Ax = b via Cholesky (LLT) decomposition.
///
/// Factors A = L * L^T, then delegates to `LLTResult::solve(b)`.
///
/// Requires A to be symmetric positive definite.
///
/// @param A  An n x n SPD matrix.
/// @param b  Right-hand side vector of length n.
/// @return   `std::expected<Vector<T,N>, SolverError>` — the solution on
///           success, or a breakdown error if A is not SPD or a zero pivot
///           is encountered during substitution.
template <concepts::Matrix ADerived, concepts::Vector BDerived>
auto llt_solve(const ADerived &A, const BDerived &b) {
  using AType = std::decay_t<ADerived>;
  using T = typename AType::value_type;
  constexpr index_type N = AType::extents_type::static_extent(0);
  using ResultVec = Vector<T, N>;
  using Result = std::expected<ResultVec, solver::SolverError>;

  auto llt_result = llt(A);
  if (!llt_result) {
    return Result{std::unexpected(std::move(llt_result.error()))};
  }

  return llt_result->solve(b);
}

} // namespace zipper::utils::decomposition

#endif
