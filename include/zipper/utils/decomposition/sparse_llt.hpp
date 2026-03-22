/// @file sparse_llt.hpp
/// @brief Sparse Cholesky (LLT) decomposition for symmetric positive definite
///        matrices.
///
/// Given a sparse symmetric positive definite (SPD) matrix A, computes:
///
///   A = L * L^T
///
/// where L is a sparse lower triangular matrix with positive diagonal entries.
/// Both input and output are stored as CSR sparse matrices.
///
/// The algorithm is an up-looking sparse Cholesky: for each row j, it
/// updates the j-th row using previously computed rows of L.
///
/// Three interfaces:
///   - `sparse_llt(A)`                  -> factorisation result.
///   - `SparseLLTResult::solve(b)`      -> solves Ax = b.
///   - `sparse_llt_solve(A, b)`         -> convenience: factorise + solve.
///
/// Complexity: O(nnz(L)^2 / n) typical for sparse Cholesky.

#if !defined(ZIPPER_UTILS_DECOMPOSITION_SPARSE_LLT_HPP)
#define ZIPPER_UTILS_DECOMPOSITION_SPARSE_LLT_HPP

#include <cmath>
#include <expected>
#include <limits>
#include <vector>

#include <zipper/COOMatrix.hpp>
#include <zipper/CSRMatrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/solver/result.hpp>

namespace zipper::utils::decomposition {

// ─────────────────────────────────────────────────────────────────────────────
// Result type
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a sparse Cholesky (LLT) decomposition.
template <typename T, index_type N> struct SparseLLTResult {
  using value_type = T;
  using matrix_type = CSRMatrix<T, N, N>;

  /// Lower triangular Cholesky factor L.
  matrix_type L;

  /// @brief Solve A*x = b using the stored sparse Cholesky factor.
  ///
  /// Performs:
  ///   1. Forward substitution:  L * y = b
  ///   2. Back substitution:     L^T * x = y
  template <concepts::Vector BDerived>
  auto solve(const BDerived &b) const
      -> std::expected<Vector<T, N>, solver::SolverError> {
    using ResultVec = Vector<T, N>;
    using Result = std::expected<ResultVec, solver::SolverError>;

    const index_type n = L.extent(0);

    // 1. Forward substitution: L * y = b
    ResultVec y(n);
    for (index_type i = 0; i < n; ++i) {
      T sum = T{0};
      for (index_type j = 0; j < i; ++j) {
        sum += L(i, j) * y(j);
      }
      T diag = L(i, i);
      if (std::abs(diag) <= std::numeric_limits<T>::epsilon()) {
        return Result{std::unexpected(solver::SolverError{
            .kind = solver::SolverError::Kind::breakdown,
            .message = "Sparse LLT solve: zero diagonal at row " +
                       std::to_string(i)})};
      }
      y(i) = (b(i) - sum) / diag;
    }

    // 2. Back substitution: L^T * x = y
    ResultVec x(n);
    for (index_type i = n; i-- > 0;) {
      T sum = T{0};
      for (index_type j = i + 1; j < n; ++j) {
        sum += L(j, i) * x(j);  // L^T(i,j) = L(j,i)
      }
      T diag = L(i, i);
      x(i) = (y(i) - sum) / diag;
    }

    return Result{std::move(x)};
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Sparse Cholesky (LLT) factorisation
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Sparse Cholesky (LLT) decomposition of a symmetric positive
///        definite matrix.
///
/// @param A  An n x n symmetric positive definite matrix.
/// @return   `std::expected<SparseLLTResult<T,N>, SolverError>`.
template <concepts::Matrix Derived>
auto sparse_llt(const Derived &A)
    -> std::expected<
        SparseLLTResult<typename std::decay_t<Derived>::value_type,
                        std::decay_t<Derived>::extents_type::static_extent(0)>,
        solver::SolverError> {
  using AType = std::decay_t<Derived>;
  using T = typename AType::value_type;
  constexpr index_type N = AType::extents_type::static_extent(0);
  using Result = std::expected<SparseLLTResult<T, N>, solver::SolverError>;

  const index_type n = A.extent(0);

  // Build L column-by-column.
  // For each column j, L(j,j) and L(i,j) for i > j.
  // We accumulate into a dense workspace, then extract non-zeros.
  COOMatrix<T, N, N> L_coo;
  if constexpr (N == dynamic_extent) {
    L_coo = COOMatrix<T, N, N>(n, n);
  }

  // Store L columns for back-reference during the up-looking phase.
  // l_col[j] = {(row, value)} for column j, sorted by row.
  std::vector<std::vector<std::pair<index_type, T>>> l_col(n);

  for (index_type j = 0; j < n; ++j) {
    // Compute L(j,j) = sqrt(A(j,j) - sum_{k<j} L(j,k)^2)
    T sum_sq = T{0};
    // Gather L(j, k) for k < j from previously computed columns
    std::vector<std::pair<index_type, T>> lj_entries; // (k, L(j,k))
    for (index_type k = 0; k < j; ++k) {
      // Check if L(j, k) exists in l_col[k]
      for (auto &[row, val] : l_col[k]) {
        if (row == j) {
          sum_sq += val * val;
          lj_entries.emplace_back(k, val);
          break;
        }
      }
    }

    T diag = A(j, j) - sum_sq;
    if (diag <= T{0}) {
      return Result{std::unexpected(solver::SolverError{
          .kind = solver::SolverError::Kind::breakdown,
          .message = "Sparse LLT: matrix is not positive definite "
                     "(non-positive diagonal at column " +
                     std::to_string(j) + ")"})};
    }
    T l_jj = std::sqrt(diag);
    L_coo.emplace(j, j) = l_jj;

    // Compute L(i, j) for i > j:
    // L(i,j) = (A(i,j) - sum_{k<j} L(i,k)*L(j,k)) / L(j,j)
    for (index_type i = j + 1; i < n; ++i) {
      T a_ij = A(i, j);
      if (a_ij == T{0}) {
        // Check if any L(i,k)*L(j,k) terms exist
        bool has_fill = false;
        for (auto &[k, ljk] : lj_entries) {
          for (auto &[row, lik] : l_col[k]) {
            if (row == i) {
              has_fill = true;
              break;
            }
          }
          if (has_fill) break;
        }
        if (!has_fill) continue; // skip — will be zero
      }

      T s = T{0};
      for (auto &[k, ljk] : lj_entries) {
        for (auto &[row, lik] : l_col[k]) {
          if (row == i) {
            s += lik * ljk;
            break;
          }
        }
      }
      T l_ij = (a_ij - s) / l_jj;
      if (std::abs(l_ij) > std::numeric_limits<T>::epsilon() * 100) {
        L_coo.emplace(i, j) = l_ij;
        l_col[j].emplace_back(i, l_ij);
      }
    }
  }

  L_coo.compress();
  return Result{SparseLLTResult<T, N>{.L = CSRMatrix<T, N, N>(L_coo)}};
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience solve
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Solve Ax = b via sparse Cholesky (LLT) decomposition.
template <concepts::Matrix ADerived, concepts::Vector BDerived>
auto sparse_llt_solve(const ADerived &A, const BDerived &b) {
  using AType = std::decay_t<ADerived>;
  using T = typename AType::value_type;
  constexpr index_type N = AType::extents_type::static_extent(0);
  using ResultVec = Vector<T, N>;
  using Result = std::expected<ResultVec, solver::SolverError>;

  auto llt_result = sparse_llt(A);
  if (!llt_result) {
    return Result{std::unexpected(std::move(llt_result.error()))};
  }

  return llt_result->solve(b);
}

} // namespace zipper::utils::decomposition

#endif
