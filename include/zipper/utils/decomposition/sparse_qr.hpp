/// @file sparse_qr.hpp
/// @brief Sparse QR decomposition via Givens rotations.
///
/// Given a sparse matrix A (m x n), computes:
///
///   A = Q * R
///
/// where Q is an m x m orthogonal matrix (stored implicitly as a sequence
/// of Givens rotations) and R is an m x n upper triangular sparse matrix.
///
/// The algorithm applies Givens rotations column by column to zero out
/// sub-diagonal entries.  Each rotation is stored as a (row_i, row_j,
/// cosine, sine) tuple.
///
/// Three interfaces:
///   - `sparse_qr(A)`                  -> factorisation result.
///   - `SparseQRResult::solve(b)`      -> solves min ||Ax - b|| (least squares).
///   - `SparseQRResult::rank(tol)`     -> numerical rank from R diagonal.
///   - `sparse_qr_solve(A, b)`         -> convenience: factorise + solve.
///
/// The rank is computed by counting non-negligible diagonal entries of R.

#if !defined(ZIPPER_UTILS_DECOMPOSITION_SPARSE_QR_HPP)
#define ZIPPER_UTILS_DECOMPOSITION_SPARSE_QR_HPP

#include <algorithm>
#include <cmath>
#include <expected>
#include <limits>
#include <vector>

#include <zipper/COOMatrix.hpp>
#include <zipper/CSRMatrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/solver/result.hpp>

namespace zipper::utils::decomposition {

namespace detail {

/// A single Givens rotation: zeroes out position (i, col) using pivot at
/// (j, col).
struct GivensRotation {
  index_type i; ///< Row to zero out (i > j)
  index_type j; ///< Pivot row
  double c;     ///< Cosine
  double s;     ///< Sine
};

} // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
// Result type
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a sparse QR decomposition.
///
/// Q is stored implicitly as a sequence of Givens rotations.  R is a sparse
/// upper triangular matrix.
template <typename T, index_type M, index_type N> struct SparseQRResult {
  using value_type = T;

  /// Givens rotations (applied in order to transform A into R).
  std::vector<detail::GivensRotation> rotations;

  /// Upper triangular factor R (m x n).
  CSRMatrix<T, M, N> R;

  /// Dimensions
  index_type m;
  index_type n;

  /// @brief Solve min ||Ax - b||_2 using the stored QR factors.
  ///
  /// Computes Q^T * b by applying stored Givens rotations, then solves
  /// the upper triangular system R * x = (Q^T b)_{1:n}.
  template <concepts::Vector BDerived>
  auto solve(const BDerived &b) const
      -> std::expected<Vector<T, N>, solver::SolverError> {
    using ResultVec = Vector<T, N>;
    using Result = std::expected<ResultVec, solver::SolverError>;

    // 1. Compute Q^T * b by applying each Givens rotation
    Vector<T, M> Qtb(m);
    for (index_type i = 0; i < m; ++i) {
      Qtb(i) = b(i);
    }
    for (const auto &g : rotations) {
      T xi = Qtb(g.j);
      T xj = Qtb(g.i);
      Qtb(g.j) = g.c * xi + g.s * xj;
      Qtb(g.i) = -g.s * xi + g.c * xj;
    }

    // 2. Back substitution on the top n x n block of R
    const index_type p = std::min(m, n);
    ResultVec x(n);
    for (index_type i = 0; i < n; ++i) {
      x(i) = T{0};
    }

    for (index_type i = p; i-- > 0;) {
      T diag = R(i, i);
      if (std::abs(diag) <= std::numeric_limits<T>::epsilon()) {
        return Result{std::unexpected(solver::SolverError{
            .kind = solver::SolverError::Kind::breakdown,
            .message = "Sparse QR solve: zero diagonal in R at row " +
                       std::to_string(i)})};
      }
      T sum = T{0};
      for (index_type j = i + 1; j < n; ++j) {
        sum += R(i, j) * x(j);
      }
      x(i) = (Qtb(i) - sum) / diag;
    }

    return Result{std::move(x)};
  }

  /// @brief Compute the numerical rank from R diagonal entries.
  ///
  /// @param tol  Tolerance.  If negative, uses n * eps * max(|R(i,i)|).
  auto rank(T tol = T{-1}) const -> index_type {
    const index_type p = std::min(m, n);
    if (tol < T{0}) {
      T max_r = T{0};
      for (index_type i = 0; i < p; ++i) {
        max_r = std::max(max_r, std::abs(R(i, i)));
      }
      tol = static_cast<T>(n) * std::numeric_limits<T>::epsilon() * max_r;
    }
    index_type r = 0;
    for (index_type i = 0; i < p; ++i) {
      if (std::abs(R(i, i)) > tol) {
        ++r;
      }
    }
    return r;
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Sparse QR factorisation
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Sparse QR decomposition via Givens rotations.
///
/// @param A  An m x n matrix.
/// @return   `std::expected<SparseQRResult<T,M,N>, SolverError>`.
template <concepts::Matrix Derived>
auto sparse_qr(const Derived &A)
    -> std::expected<
        SparseQRResult<typename std::decay_t<Derived>::value_type,
                       std::decay_t<Derived>::extents_type::static_extent(0),
                       std::decay_t<Derived>::extents_type::static_extent(1)>,
        solver::SolverError> {
  using AType = std::decay_t<Derived>;
  using T = typename AType::value_type;
  constexpr index_type M = AType::extents_type::static_extent(0);
  constexpr index_type N = AType::extents_type::static_extent(1);
  using Result = std::expected<SparseQRResult<T, M, N>, solver::SolverError>;

  const index_type m = A.extent(0);
  const index_type n = A.extent(1);

  // Work on a dense copy of A for the rotation process.
  // For truly large sparse matrices, one would maintain sparse row
  // structures, but this is correct and handles moderate sizes well.
  std::vector<std::vector<T>> rows(m, std::vector<T>(n, T{0}));
  for (index_type i = 0; i < m; ++i) {
    for (index_type j = 0; j < n; ++j) {
      rows[i][j] = A(i, j);
    }
  }

  std::vector<detail::GivensRotation> rotations;
  const index_type p = std::min(m, n);

  for (index_type col = 0; col < p; ++col) {
    // Zero out sub-diagonal entries in this column
    for (index_type row = m - 1; row > col; --row) {
      if (std::abs(rows[row][col]) <= std::numeric_limits<T>::epsilon()) {
        continue;
      }

      // Compute Givens rotation to zero out rows[row][col] using
      // rows[col][col] as pivot
      T a = rows[col][col];
      T b = rows[row][col];
      T r = std::hypot(a, b);
      T c = a / r;
      T s = b / r;

      rotations.push_back(
          {.i = row, .j = col, .c = static_cast<double>(c),
           .s = static_cast<double>(s)});

      // Apply rotation to rows col and row
      for (index_type j = col; j < n; ++j) {
        T x = rows[col][j];
        T y = rows[row][j];
        rows[col][j] = c * x + s * y;
        rows[row][j] = -s * x + c * y;
      }
    }
  }

  // Build sparse R from the rotated rows
  COOMatrix<T, M, N> R_coo;
  if constexpr (M == dynamic_extent || N == dynamic_extent) {
    R_coo = COOMatrix<T, M, N>(m, n);
  }
  for (index_type i = 0; i < m; ++i) {
    for (index_type j = i; j < n; ++j) {
      if (std::abs(rows[i][j]) > std::numeric_limits<T>::epsilon() * 100) {
        R_coo.emplace(i, j) = rows[i][j];
      }
    }
  }
  R_coo.compress();

  return Result{SparseQRResult<T, M, N>{
      .rotations = std::move(rotations),
      .R = CSRMatrix<T, M, N>(R_coo),
      .m = m,
      .n = n}};
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience solve
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Solve min ||Ax - b||_2 via sparse QR decomposition.
template <concepts::Matrix ADerived, concepts::Vector BDerived>
auto sparse_qr_solve(const ADerived &A, const BDerived &b) {
  using AType = std::decay_t<ADerived>;
  using T = typename AType::value_type;
  constexpr index_type N = AType::extents_type::static_extent(1);
  using ResultVec = Vector<T, N>;
  using Result = std::expected<ResultVec, solver::SolverError>;

  auto qr_result = sparse_qr(A);
  if (!qr_result) {
    return Result{std::unexpected(std::move(qr_result.error()))};
  }

  return qr_result->solve(b);
}

} // namespace zipper::utils::decomposition

#endif
