/// @file lu.hpp
/// @brief PLU decomposition (LU with partial pivoting) for general square
///        matrices.
///
/// Given a square matrix A, the PLU decomposition factors it as:
///
///   P * A = L * U
///
/// where P is a permutation matrix, L is a unit lower triangular matrix (ones
/// on the diagonal), and U is an upper triangular matrix.  The permutation P
/// encodes the row swaps performed during Gaussian elimination with partial
/// pivoting (selecting the largest-magnitude pivot element in each column).
///
/// The factored form stores L and U in a single combined matrix: the strict
/// lower triangle holds the multipliers of L (with implicit unit diagonal),
/// and the upper triangle (including the diagonal) holds U.  This matches the
/// standard LAPACK convention.
///
/// The permutation is stored as a DataArray `perm` where `perm(i)` is the
/// original row index of the row currently in position i after all swaps.
/// A sign integer tracks the parity of the permutation (for computing
/// determinants).
///
/// Three interfaces:
///   - `plu(A)`          -- returns the PLU factorisation.
///   - `PLUResult::solve(b)` -- solves Ax = b using the stored factorisation.
///   - `plu_solve(A, b)` -- convenience: factorise then solve in one call.
///
/// The solve proceeds in three stages:
///   1. Permute:             Pb = P * b
///   2. Forward substitution: L * y = Pb   (unit lower triangular)
///   3. Back substitution:    U * x = y    (upper triangular)
///
/// Singularity is detected when a zero (or near-zero) pivot is encountered
/// during factorisation.  The functions return `std::expected` with a
/// `SolverError` on failure.
///
/// Complexity: O(2n^3/3) for the factorisation, O(n^2) for each solve.

#if !defined(ZIPPER_UTILS_DECOMPOSITION_LU_HPP)
#define ZIPPER_UTILS_DECOMPOSITION_LU_HPP

#include <algorithm>
#include <cmath>
#include <expected>
#include <limits>
#include <numeric>

#include <zipper/DataArray.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/unary/TriangularView.hpp>
#include <zipper/utils/solver/result.hpp>

namespace zipper::utils::decomposition {

// ─────────────────────────────────────────────────────────────────────────────
// Result type
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a PLU decomposition.
///
/// `LU` is an n x n matrix whose lower triangle stores the multipliers of L
/// (with an implicit unit diagonal) and whose upper triangle (including the
/// diagonal) stores U.  `perm` is the row permutation vector.  `sign` is +1
/// or -1 depending on the parity of the permutation (useful for determinants).
///
/// Calling `.solve(b)` performs permuted forward/back substitution using the
/// stored factors to solve A*x = b without re-factoring.
template <typename T, index_type N> struct PLUResult {
  /// Scalar type of the decomposition.
  using value_type = T;

  /// Combined LU matrix (lower triangle = L multipliers, upper triangle = U).
  Matrix<T, N, N> LU;
  /// Row permutation: perm[i] = original row index at position i.
  DataArray<index_type, N> perm;
  /// Sign of the permutation (+1 or -1).
  int sign;

  /// @brief Unit lower triangular factor L.
  ///
  /// Returns a read-only TriangularView of the combined LU matrix with
  /// UnitLower mode: the strict lower triangle contains the L multipliers,
  /// and the diagonal is implicitly 1.
  auto L() const {
    return LU.template as_triangular<expression::TriangularMode::UnitLower>();
  }

  /// @brief Upper triangular factor U.
  ///
  /// Returns a read-only TriangularView of the combined LU matrix with
  /// Upper mode: the upper triangle (including the diagonal) contains U.
  auto U() const {
    return LU.template as_triangular<expression::TriangularMode::Upper>();
  }

  /// @brief Solve A*x = b using the stored PLU factors.
  ///
  /// Performs:
  ///   1. Apply permutation to b:  Pb(i) = b(perm(i))
  ///   2. Forward substitution:    L * y = Pb  (unit lower triangular)
  ///   3. Back substitution:       U * x = y   (upper triangular)
  ///
  /// @param b  Right-hand side vector of length n.
  /// @return   `std::expected<Vector<T,N>, SolverError>` — the solution on
  ///           success, or a breakdown error if a zero pivot is encountered.
  template <concepts::Vector BDerived>
  auto solve(const BDerived &b) const
      -> std::expected<Vector<T, N>, solver::SolverError> {
    using ResultVec = Vector<T, N>;
    using Result = std::expected<ResultVec, solver::SolverError>;

    const index_type n = LU.rows();

    // 1. Apply permutation: Pb(i) = b(perm(i)).
    ResultVec Pb(n);
    for (index_type i = 0; i < n; ++i) {
      Pb(i) = b(perm(i));
    }

    // 2. Forward substitution: L * y = Pb  (L is unit lower triangular).
    auto y_result = L().solve(Pb);

    if (!y_result) {
      return Result{std::unexpected(std::move(y_result.error()))};
    }

    // 3. Back substitution: U * x = y  (U is upper triangular).
    auto x_result = U().solve(*y_result);

    if (!x_result) {
      return Result{std::unexpected(std::move(x_result.error()))};
    }

    return Result{std::move(*x_result)};
  }

  /// @brief Compute the determinant of the original matrix A.
  ///
  /// det(A) = sign * product of U diagonal entries.
  auto determinant() const -> T {
    const index_type n = LU.rows();
    T det = static_cast<T>(sign);
    for (index_type i = 0; i < n; ++i) {
      det *= LU(i, i);
    }
    return det;
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// PLU factorisation
// ─────────────────────────────────────────────────────────────────────────────

/// @brief PLU decomposition of a general square matrix.
///
/// Performs Gaussian elimination with partial pivoting (largest-magnitude
/// pivot in each column).
///
/// @param A  An n x n matrix.
/// @return   `std::expected<PLUResult<T,N>, SolverError>` — the factors on
///           success, or a breakdown error if the matrix is singular (zero
///           pivot encountered).
template <concepts::Matrix Derived>
auto plu(const Derived &A)
    -> std::expected<
        PLUResult<typename std::decay_t<Derived>::value_type,
                  std::decay_t<Derived>::extents_type::static_extent(0)>,
        solver::SolverError> {
  using AType = std::decay_t<Derived>;
  using T = typename AType::value_type;
  constexpr index_type N = AType::extents_type::static_extent(0);
  using Result = std::expected<PLUResult<T, N>, solver::SolverError>;

  const index_type n = A.rows();

  // Copy A into the combined LU matrix via converting constructor.
  Matrix<T, N, N> LU(A);

  // Initialise the permutation vector to the identity.
  DataArray<index_type, N> perm(n);
  std::iota(perm.begin(), perm.end(), index_type{0});
  int sign = 1;

  for (index_type k = 0; k < n; ++k) {
    // Partial pivoting: find the row with the largest absolute value in
    // column k (from row k downward).
    index_type max_row = k;
    T max_val = std::abs(LU(k, k));
    for (index_type i = k + 1; i < n; ++i) {
      T val = std::abs(LU(i, k));
      if (val > max_val) {
        max_val = val;
        max_row = i;
      }
    }

    // Check for singular matrix.
    if (max_val <= std::numeric_limits<T>::epsilon()) {
      return Result{std::unexpected(solver::SolverError{
          .kind = solver::SolverError::Kind::breakdown,
          .message =
              "PLU decomposition: matrix is singular (zero pivot at column " +
              std::to_string(k) + ")"})};
    }

    // Swap rows if necessary.
    if (max_row != k) {
      // Swap entire rows in LU (includes both L multipliers and U entries)
      // via row views.  A temporary is needed to avoid aliasing.
      Vector<T, N> tmp(LU.row(k));
      LU.row(k) = LU.row(max_row);
      LU.row(max_row) = tmp;
      std::swap(perm(k), perm(max_row));
      sign = -sign;
    }

    // Gaussian elimination: compute multipliers and update trailing submatrix.
    T pivot = LU(k, k);
    for (index_type i = k + 1; i < n; ++i) {
      LU(i, k) /= pivot; // Store the multiplier in the lower triangle.
      for (index_type j = k + 1; j < n; ++j) {
        LU(i, j) -= LU(i, k) * LU(k, j);
      }
    }
  }

  return Result{PLUResult<T, N>{
      .LU = std::move(LU), .perm = std::move(perm), .sign = sign}};
}

// ─────────────────────────────────────────────────────────────────────────────
// PLU solve
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Solve Ax = b via PLU decomposition.
///
/// Factors P*A = L*U, then delegates to `PLUResult::solve(b)`.
///
/// @param A  An n x n matrix.
/// @param b  Right-hand side vector of length n.
/// @return   `std::expected<Vector<T,N>, SolverError>` — the solution on
///           success, or a breakdown error if the matrix is singular.
template <concepts::Matrix ADerived, concepts::Vector BDerived>
auto plu_solve(const ADerived &A, const BDerived &b) {
  using AType = std::decay_t<ADerived>;
  using T = typename AType::value_type;
  constexpr index_type N = AType::extents_type::static_extent(0);
  using ResultVec = Vector<T, N>;
  using Result = std::expected<ResultVec, solver::SolverError>;

  auto plu_result = plu(A);
  if (!plu_result) {
    return Result{std::unexpected(std::move(plu_result.error()))};
  }

  return plu_result->solve(b);
}

} // namespace zipper::utils::decomposition

#endif
