/// @file sparse_ldlt.hpp
/// @brief Sparse LDLT decomposition for symmetric positive semi-definite
///        matrices.
///
/// Given a sparse symmetric matrix A, computes:
///
///   A = L * D * L^T
///
/// where L is a sparse unit lower triangular matrix and D is a diagonal
/// matrix.  Both L and D are stored sparsely.
///
/// Compared to sparse LLT, LDLT avoids square roots and handles positive
/// semi-definite matrices (where D may have zero entries).
///
/// Four interfaces:
///   - `sparse_ldlt(A)`                   -> factorisation result.
///   - `SparseLDLTResult::solve(b)`       -> solves Ax = b.
///   - `SparseLDLTResult::rank(tol)`      -> numerical rank from D.
///   - `sparse_ldlt_solve(A, b)`          -> convenience: factorise + solve.
///
/// Complexity: depends on fill-in.

#if !defined(ZIPPER_UTILS_DECOMPOSITION_SPARSE_LDLT_HPP)
#define ZIPPER_UTILS_DECOMPOSITION_SPARSE_LDLT_HPP

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

/// Result of a sparse LDLT decomposition.
template <typename T, index_type N> struct SparseLDLTResult {
  using value_type = T;
  using matrix_type = CSRMatrix<T, N, N>;

  /// Unit lower triangular factor L.
  matrix_type L;
  /// Diagonal factor D (stored as a dense vector).
  Vector<T, N> D;

  /// @brief Solve A*x = b using the stored sparse LDLT factors.
  ///
  /// Performs:
  ///   1. Forward substitution:  L * y = b      (unit lower triangular)
  ///   2. Diagonal solve:        D * z = y
  ///   3. Back substitution:     L^T * x = z    (unit upper triangular)
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
      y(i) = b(i) - sum;
    }

    // 2. Diagonal solve: z(i) = y(i) / D(i)
    ResultVec z(n);
    for (index_type i = 0; i < n; ++i) {
      if (std::abs(D(i)) <= std::numeric_limits<T>::epsilon()) {
        return Result{std::unexpected(solver::SolverError{
            .kind = solver::SolverError::Kind::breakdown,
            .message = "Sparse LDLT solve: zero diagonal D(" +
                       std::to_string(i) + ")"})};
      }
      z(i) = y(i) / D(i);
    }

    // 3. Back substitution: L^T * x = z
    ResultVec x(n);
    for (index_type i = n; i-- > 0;) {
      T sum = T{0};
      for (index_type j = i + 1; j < n; ++j) {
        sum += L(j, i) * x(j);  // L^T(i,j) = L(j,i)
      }
      x(i) = z(i) - sum;
    }

    return Result{std::move(x)};
  }

  /// @brief Compute the numerical rank of A from the diagonal D.
  ///
  /// Counts the number of diagonal entries whose absolute value exceeds
  /// the tolerance.
  ///
  /// @param tol  Tolerance threshold.  If negative, uses
  ///             n * eps * max(|D(i)|).
  /// @return     The numerical rank.
  auto rank(T tol = T{-1}) const -> index_type {
    const index_type n = D.extent(0);
    if (tol < T{0}) {
      T max_d = T{0};
      for (index_type i = 0; i < n; ++i) {
        max_d = std::max(max_d, std::abs(D(i)));
      }
      tol = static_cast<T>(n) * std::numeric_limits<T>::epsilon() * max_d;
    }
    index_type r = 0;
    for (index_type i = 0; i < n; ++i) {
      if (std::abs(D(i)) > tol) {
        ++r;
      }
    }
    return r;
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Sparse LDLT factorisation
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Sparse LDLT decomposition of a symmetric matrix.
///
/// @param A  An n x n symmetric matrix.
/// @return   `std::expected<SparseLDLTResult<T,N>, SolverError>`.
template <concepts::Matrix Derived>
auto sparse_ldlt(const Derived &A)
    -> std::expected<
        SparseLDLTResult<typename std::decay_t<Derived>::value_type,
                         std::decay_t<Derived>::extents_type::static_extent(0)>,
        solver::SolverError> {
  using AType = std::decay_t<Derived>;
  using T = typename AType::value_type;
  constexpr index_type N = AType::extents_type::static_extent(0);
  using Result = std::expected<SparseLDLTResult<T, N>, solver::SolverError>;

  const index_type n = A.extent(0);

  COOMatrix<T, N, N> L_coo;
  if constexpr (N == dynamic_extent) {
    L_coo = COOMatrix<T, N, N>(n, n);
  }

  Vector<T, N> D_vec(n);
  for (index_type i = 0; i < n; ++i) {
    D_vec(i) = T{0};
  }

  // Store L column entries for back-reference.
  // l_col[k] = {(row, value)} for column k, rows > k.
  std::vector<std::vector<std::pair<index_type, T>>> l_col(n);

  for (index_type j = 0; j < n; ++j) {
    // D(j) = A(j,j) - sum_{k<j} L(j,k)^2 * D(k)
    T sum = T{0};
    std::vector<std::pair<index_type, T>> lj_entries; // (k, L(j,k)) for k < j
    for (index_type k = 0; k < j; ++k) {
      for (auto &[row, val] : l_col[k]) {
        if (row == j) {
          sum += val * val * D_vec(k);
          lj_entries.emplace_back(k, val);
          break;
        }
      }
    }
    D_vec(j) = A(j, j) - sum;

    // Unit diagonal in L
    L_coo.emplace(j, j) = T{1};

    if (std::abs(D_vec(j)) <= std::numeric_limits<T>::epsilon()) {
      // Zero pivot — PSD: skip sub-diagonal computation for this column
      continue;
    }

    // L(i, j) = (A(i,j) - sum_{k<j} L(i,k)*L(j,k)*D(k)) / D(j)
    for (index_type i = j + 1; i < n; ++i) {
      T a_ij = A(i, j);

      // Check if there are any fill-in terms
      bool has_terms = (a_ij != T{0});
      if (!has_terms) {
        for (auto &[k, ljk] : lj_entries) {
          for (auto &[row, lik] : l_col[k]) {
            if (row == i) {
              has_terms = true;
              break;
            }
          }
          if (has_terms) break;
        }
        if (!has_terms) continue;
      }

      T s = T{0};
      for (auto &[k, ljk] : lj_entries) {
        for (auto &[row, lik] : l_col[k]) {
          if (row == i) {
            s += lik * ljk * D_vec(k);
            break;
          }
        }
      }
      T l_ij = (a_ij - s) / D_vec(j);
      if (std::abs(l_ij) > std::numeric_limits<T>::epsilon() * 100) {
        L_coo.emplace(i, j) = l_ij;
        l_col[j].emplace_back(i, l_ij);
      }
    }
  }

  L_coo.compress();
  return Result{SparseLDLTResult<T, N>{
      .L = CSRMatrix<T, N, N>(L_coo), .D = std::move(D_vec)}};
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience solve
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Solve Ax = b via sparse LDLT decomposition.
template <concepts::Matrix ADerived, concepts::Vector BDerived>
auto sparse_ldlt_solve(const ADerived &A, const BDerived &b) {
  using AType = std::decay_t<ADerived>;
  using T = typename AType::value_type;
  constexpr index_type N = AType::extents_type::static_extent(0);
  using ResultVec = Vector<T, N>;
  using Result = std::expected<ResultVec, solver::SolverError>;

  auto ldlt_result = sparse_ldlt(A);
  if (!ldlt_result) {
    return Result{std::unexpected(std::move(ldlt_result.error()))};
  }

  return ldlt_result->solve(b);
}

} // namespace zipper::utils::decomposition

#endif
