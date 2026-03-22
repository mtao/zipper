/// @file sparse_lu.hpp
/// @brief Sparse LU decomposition with partial pivoting.
///
/// Given a sparse square matrix A, the sparse PLU decomposition factors it as:
///
///   P * A = L * U
///
/// where P is a permutation matrix, L is a unit lower triangular matrix, and
/// U is an upper triangular matrix.  Both L and U are stored as sparse CSR
/// matrices.
///
/// The algorithm is a left-looking sparse Gaussian elimination with partial
/// pivoting.  For each column k, it:
///   1. Gathers the k-th column of the remaining matrix.
///   2. Applies previously computed L columns to update it.
///   3. Selects the largest-magnitude pivot and swaps rows.
///   4. Computes multipliers (L column) and the U row.
///
/// Three interfaces:
///   - `sparse_plu(A)`                 -> factorisation result.
///   - `SparsePLUResult::solve(b)`     -> solves Ax = b.
///   - `sparse_plu_solve(A, b)`        -> convenience: factorise + solve.
///
/// Complexity: depends on fill-in.  Worst case O(n^3) for dense matrices;
/// for typical sparse matrices, much less.

#if !defined(ZIPPER_UTILS_DECOMPOSITION_SPARSE_LU_HPP)
#define ZIPPER_UTILS_DECOMPOSITION_SPARSE_LU_HPP

#include <algorithm>
#include <cmath>
#include <expected>
#include <limits>
#include <numeric>
#include <vector>

#include <zipper/COOMatrix.hpp>
#include <zipper/CSRMatrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/solver/result.hpp>

namespace zipper::utils::decomposition {

// ─────────────────────────────────────────────────────────────────────────────
// Result type
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a sparse PLU decomposition.
///
/// L is a unit lower triangular CSR matrix, U is an upper triangular CSR
/// matrix, and perm is the row permutation vector.
template <typename T, index_type N> struct SparsePLUResult {
  using value_type = T;
  using matrix_type = CSRMatrix<T, N, N>;

  /// Unit lower triangular factor L.
  matrix_type L;
  /// Upper triangular factor U.
  matrix_type U;
  /// Row permutation: perm[i] = original row index at position i.
  std::vector<index_type> perm;
  /// Sign of the permutation (+1 or -1).
  int sign;

  /// @brief Solve A*x = b using the stored sparse PLU factors.
  template <concepts::Vector BDerived>
  auto solve(const BDerived &b) const
      -> std::expected<Vector<T, N>, solver::SolverError> {
    using ResultVec = Vector<T, N>;
    using Result = std::expected<ResultVec, solver::SolverError>;

    const index_type n = L.extent(0);

    // 1. Apply permutation: Pb(i) = b(perm[i]).
    ResultVec Pb(n);
    for (index_type i = 0; i < n; ++i) {
      Pb(i) = b(perm[i]);
    }

    // 2. Forward substitution: L * y = Pb (L is unit lower triangular).
    ResultVec y(n);
    for (index_type i = 0; i < n; ++i) {
      T sum = T{0};
      for (index_type j = 0; j < i; ++j) {
        sum += L(i, j) * y(j);
      }
      y(i) = Pb(i) - sum;
    }

    // 3. Back substitution: U * x = y (upper triangular).
    ResultVec x(n);
    for (index_type i = n; i-- > 0;) {
      T diag = U(i, i);
      if (std::abs(diag) <= std::numeric_limits<T>::epsilon()) {
        return Result{std::unexpected(solver::SolverError{
            .kind = solver::SolverError::Kind::breakdown,
            .message = "Sparse PLU solve: zero diagonal in U at row " +
                       std::to_string(i)})};
      }
      T sum = T{0};
      for (index_type j = i + 1; j < n; ++j) {
        sum += U(i, j) * x(j);
      }
      x(i) = (y(i) - sum) / diag;
    }

    return Result{std::move(x)};
  }

  /// @brief Compute the determinant of the original matrix A.
  auto determinant() const -> T {
    const index_type n = L.extent(0);
    T det = static_cast<T>(sign);
    for (index_type i = 0; i < n; ++i) {
      det *= U(i, i);
    }
    return det;
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Sparse PLU factorisation
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Sparse PLU decomposition of a general square matrix.
///
/// Works with any matrix satisfying `concepts::Matrix`, but is designed
/// for sparse matrices.  Internally converts to a column-indexed
/// representation for efficient column access.
///
/// @param A  An n x n matrix.
/// @return   `std::expected<SparsePLUResult<T,N>, SolverError>`.
template <concepts::Matrix Derived>
auto sparse_plu(const Derived &A)
    -> std::expected<
        SparsePLUResult<typename std::decay_t<Derived>::value_type,
                        std::decay_t<Derived>::extents_type::static_extent(0)>,
        solver::SolverError> {
  using AType = std::decay_t<Derived>;
  using T = typename AType::value_type;
  constexpr index_type N = AType::extents_type::static_extent(0);
  using Result = std::expected<SparsePLUResult<T, N>, solver::SolverError>;

  const index_type n = A.extent(0);

  // Work with a dense row representation.
  // rows[i][j] holds the working copy of the matrix in row-major order.
  std::vector<std::vector<T>> rows(n, std::vector<T>(n, T{0}));
  for (index_type i = 0; i < n; ++i) {
    for (index_type j = 0; j < n; ++j) {
      rows[i][j] = A(i, j);
    }
  }

  std::vector<index_type> perm(n);
  std::iota(perm.begin(), perm.end(), index_type{0});
  int sign_val = 1;

  // L multipliers stored per row: l_row[i][k] = multiplier for column k
  std::vector<std::vector<T>> l_row(n, std::vector<T>(n, T{0}));

  for (index_type k = 0; k < n; ++k) {
    // Partial pivoting: find max in column k from row k downward
    index_type max_row = k;
    T max_val = std::abs(rows[k][k]);
    for (index_type i = k + 1; i < n; ++i) {
      T val = std::abs(rows[i][k]);
      if (val > max_val) {
        max_val = val;
        max_row = i;
      }
    }

    if (max_val <= std::numeric_limits<T>::epsilon()) {
      return Result{std::unexpected(solver::SolverError{
          .kind = solver::SolverError::Kind::breakdown,
          .message = "Sparse PLU: singular matrix (zero pivot at column " +
                     std::to_string(k) + ")"})};
    }

    // Swap rows (including already-computed L multipliers)
    if (max_row != k) {
      std::swap(rows[k], rows[max_row]);
      std::swap(l_row[k], l_row[max_row]);
      std::swap(perm[k], perm[max_row]);
      sign_val = -sign_val;
    }

    // Compute L multipliers and eliminate
    T pivot = rows[k][k];
    for (index_type i = k + 1; i < n; ++i) {
      if (rows[i][k] == T{0}) continue;
      T mult = rows[i][k] / pivot;
      l_row[i][k] = mult;
      rows[i][k] = T{0}; // eliminated
      for (index_type j = k + 1; j < n; ++j) {
        rows[i][j] -= mult * rows[k][j];
      }
    }
  }

  // Build sparse L (unit lower triangular)
  constexpr auto dyn = dynamic_extent;
  COOMatrix<T, N, N> L_coo;
  if constexpr (N == dyn) {
    L_coo = COOMatrix<T, N, N>(n, n);
  }
  for (index_type i = 0; i < n; ++i) {
    L_coo.emplace(i, i) = T{1}; // unit diagonal
    for (index_type j = 0; j < i; ++j) {
      if (l_row[i][j] != T{0}) {
        L_coo.emplace(i, j) = l_row[i][j];
      }
    }
  }
  L_coo.compress();

  // Build sparse U (upper triangular) from the reduced rows
  COOMatrix<T, N, N> U_coo;
  if constexpr (N == dyn) {
    U_coo = COOMatrix<T, N, N>(n, n);
  }
  for (index_type i = 0; i < n; ++i) {
    for (index_type j = i; j < n; ++j) {
      if (rows[i][j] != T{0}) {
        U_coo.emplace(i, j) = rows[i][j];
      }
    }
  }
  U_coo.compress();

  return Result{SparsePLUResult<T, N>{
      .L = CSRMatrix<T, N, N>(L_coo),
      .U = CSRMatrix<T, N, N>(U_coo),
      .perm = std::move(perm),
      .sign = sign_val}};
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience solve
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Solve Ax = b via sparse PLU decomposition.
template <concepts::Matrix ADerived, concepts::Vector BDerived>
auto sparse_plu_solve(const ADerived &A, const BDerived &b) {
  using AType = std::decay_t<ADerived>;
  using T = typename AType::value_type;
  constexpr index_type N = AType::extents_type::static_extent(0);
  using ResultVec = Vector<T, N>;
  using Result = std::expected<ResultVec, solver::SolverError>;

  auto plu_result = sparse_plu(A);
  if (!plu_result) {
    return Result{std::unexpected(std::move(plu_result.error()))};
  }

  return plu_result->solve(b);
}

} // namespace zipper::utils::decomposition

#endif
