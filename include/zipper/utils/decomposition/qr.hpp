/// @file qr.hpp
/// @brief QR decomposition via Householder reflections and Gram-Schmidt.
///
/// Three variants of QR factorisation are provided:
///
///   1. `qr(A)` — Reduced (thin) Householder QR.
///      Given an m x n matrix A with m >= n, computes Q (m x n) and R (n x n)
///      such that A = Q * R, where Q has orthonormal columns and R is upper
///      triangular.  For m < n the factorisation is still valid: Q is m x m
///      and R is m x n.
///
///   2. `qr_full(A)` — Full Householder QR.
///      Computes Q (m x m, orthogonal) and R (m x n, upper trapezoidal) such
///      that A = Q * R.  The extra m - min(m,n) columns of Q span the
///      orthogonal complement of col(A).
///
///   3. `qr_gram_schmidt(A)` — Reduced QR via classical Gram-Schmidt.
///      Uses the existing `orthogonalization::gram_schmidt()` to orthonormalise
///      the columns of A, producing Q (m x n), then computes R = Q^T * A using
///      element-level loops (since zipper may not support matrix-matrix
///      multiplication directly).  Requires m >= n.  This variant is simpler
///      but less numerically stable than Householder.
///
/// All three return result structs containing Q and R.
///
/// The Householder algorithm works by successively zeroing out the sub-diagonal
/// entries of each column of A using rank-1 reflections:
///
///   H_k = I - 2 * v_k * v_k^T
///
/// where v_k is chosen so that H_k maps the k-th column (below the diagonal)
/// to a multiple of e_1.  The product H_1 * H_2 * ... * H_p (with p =
/// min(m,n)) is an orthogonal matrix, and the transformed A is upper
/// triangular (or trapezoidal if m < n).
///
/// Rather than forming the full Householder matrices, the implementation
/// applies reflections in-place using the standard formula:
///
///   A := A - 2 * v * (v^T * A)     (for columns j+1..n-1)
///
/// and accumulates Q by applying reflections to an identity-initialised matrix.

#if !defined(ZIPPER_UTILS_DECOMPOSITION_QR_HPP)
#define ZIPPER_UTILS_DECOMPOSITION_QR_HPP

#include <algorithm>
#include <cmath>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/utils/orthogonalization/gram_schmidt.hpp>

namespace zipper::utils::decomposition {

// ─────────────────────────────────────────────────────────────────────────────
// Helper: compile-time min of two extents (respects dynamic_extent).
// ─────────────────────────────────────────────────────────────────────────────

namespace detail {

/// Compute min(A, B) at compile time, treating dynamic_extent as "unknown".
/// If either is dynamic_extent, the result is dynamic_extent (unknown at
/// compile time).
consteval index_type min_extent(index_type A, index_type B) {
  if (A == dynamic_extent || B == dynamic_extent)
    return dynamic_extent;
  return (A < B) ? A : B;
}

} // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
// Result types
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a reduced (thin) QR decomposition.
///
/// Q is m x p and R is p x n, where p = min(m, n).
template <typename T, index_type M, index_type N> struct QRReducedResult {
  static constexpr index_type P = detail::min_extent(M, N);

  /// Orthonormal matrix Q (m x p).
  Matrix<T, M, P> Q;
  /// Upper triangular matrix R (p x n).
  Matrix<T, P, N> R;
};

/// Result of a full QR decomposition.
///
/// Q is m x m (orthogonal) and R is m x n (upper trapezoidal).
template <typename T, index_type M, index_type N> struct QRFullResult {
  /// Orthogonal matrix Q (m x m).
  Matrix<T, M, M> Q;
  /// Upper trapezoidal matrix R (m x n).
  Matrix<T, M, N> R;
};

// ─────────────────────────────────────────────────────────────────────────────
// Householder QR (reduced / thin)
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Reduced (thin) QR decomposition via Householder reflections.
///
/// @param A  An m x n matrix.
/// @return   A `QRReducedResult` with Q (m x min(m,n)) and R (min(m,n) x n).
///
/// The algorithm applies p = min(m, n) Householder reflections to reduce A to
/// upper triangular form, accumulating the reflections into Q.
template <concepts::Matrix Derived> auto qr(const Derived &A) {
  using AType = std::decay_t<Derived>;
  using T = typename AType::value_type;
  constexpr index_type M = AType::extents_type::static_extent(0);
  constexpr index_type N = AType::extents_type::static_extent(1);
  constexpr index_type P = detail::min_extent(M, N);
  using Vec = Vector<T, M>;

  const index_type m = A.extent(0);
  const index_type n = A.extent(1);
  const index_type p = std::min(m, n);

  // Work on a copy of A that will be transformed into R.
  // We only need a p x n result for R, but it's easier to work on an m x n
  // copy and then extract the top p rows at the end.
  Matrix<T, M, N> R_work(A);

  // Q_full accumulates the Householder reflections.  Start as identity.
  // For the reduced QR we only need the first p columns, but we accumulate
  // into m x m and extract afterwards.
  Matrix<T, M, M> Q_full(m, m);
  Q_full = expression::nullary::Constant(T{0}, Q_full.extents());
  for (index_type i = 0; i < m; ++i) {
    Q_full(i, i) = T{1};
  }

  for (index_type k = 0; k < p; ++k) {
    // Extract the sub-column R_work(k:m-1, k) into a temporary vector.
    // We work with a full-length vector but only the entries k..m-1 are
    // non-zero in the Householder vector.
    Vec v(m);
    v = expression::nullary::Constant(T{0}, v.extents());

    T norm_sub = T{0};
    for (index_type i = k; i < m; ++i) {
      v(i) = R_work(i, k);
      norm_sub += v(i) * v(i);
    }
    norm_sub = std::sqrt(norm_sub);

    if (norm_sub < std::numeric_limits<T>::min()) {
      // Column is already zero below the diagonal — skip this reflection.
      continue;
    }

    // Choose the sign to avoid cancellation: v(k) += sign(v(k)) * ||x||.
    // If v(k) >= 0, add norm_sub; if v(k) < 0, subtract norm_sub.
    if (v(k) >= T{0}) {
      v(k) += norm_sub;
    } else {
      v(k) -= norm_sub;
    }

    // Normalise v so that the reflection is H = I - 2*v*v^T.
    T v_norm = T{0};
    for (index_type i = k; i < m; ++i) {
      v_norm += v(i) * v(i);
    }
    v_norm = std::sqrt(v_norm);
    for (index_type i = k; i < m; ++i) {
      v(i) /= v_norm;
    }

    // Apply H to R_work: R_work := R_work - 2 * v * (v^T * R_work)
    // For each column j of R_work, compute dot = v^T * R_work(:, j),
    // then R_work(i, j) -= 2 * v(i) * dot.
    for (index_type j = k; j < n; ++j) {
      T dot = T{0};
      for (index_type i = k; i < m; ++i) {
        dot += v(i) * R_work(i, j);
      }
      for (index_type i = k; i < m; ++i) {
        R_work(i, j) -= T{2} * v(i) * dot;
      }
    }

    // Apply H to Q_full: Q_full := Q_full - 2 * (Q_full * v) * v^T
    // For each row i of Q_full, compute dot = Q_full(i, :) . v,
    // then Q_full(i, j) -= 2 * dot * v(j).
    for (index_type i = 0; i < m; ++i) {
      T dot = T{0};
      for (index_type j = k; j < m; ++j) {
        dot += Q_full(i, j) * v(j);
      }
      for (index_type j = k; j < m; ++j) {
        Q_full(i, j) -= T{2} * dot * v(j);
      }
    }
  }

  // Extract the reduced Q (first p columns) and R (top p rows).
  Matrix<T, M, P> Q(m, p);
  Matrix<T, P, N> R(p, n);

  for (index_type i = 0; i < m; ++i) {
    for (index_type j = 0; j < p; ++j) {
      Q(i, j) = Q_full(i, j);
    }
  }

  for (index_type i = 0; i < p; ++i) {
    for (index_type j = 0; j < n; ++j) {
      R(i, j) = R_work(i, j);
    }
  }

  return QRReducedResult<T, M, N>{.Q = std::move(Q), .R = std::move(R)};
}

// ─────────────────────────────────────────────────────────────────────────────
// Householder QR (full)
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Full QR decomposition via Householder reflections.
///
/// @param A  An m x n matrix.
/// @return   A `QRFullResult` with Q (m x m, orthogonal) and R (m x n,
///           upper trapezoidal).
///
/// This is the same algorithm as the reduced QR, but Q is returned in full
/// (m x m) rather than truncated to the first min(m,n) columns, and R is
/// m x n rather than min(m,n) x n.
template <concepts::Matrix Derived> auto qr_full(const Derived &A) {
  using AType = std::decay_t<Derived>;
  using T = typename AType::value_type;
  constexpr index_type M = AType::extents_type::static_extent(0);
  constexpr index_type N = AType::extents_type::static_extent(1);
  using Vec = Vector<T, M>;

  const index_type m = A.extent(0);
  const index_type n = A.extent(1);
  const index_type p = std::min(m, n);

  // Work copy of A → becomes R.
  Matrix<T, M, N> R(A);

  // Accumulate Q starting from identity.
  Matrix<T, M, M> Q(m, m);
  Q = expression::nullary::Constant(T{0}, Q.extents());
  for (index_type i = 0; i < m; ++i) {
    Q(i, i) = T{1};
  }

  for (index_type k = 0; k < p; ++k) {
    Vec v(m);
    v = expression::nullary::Constant(T{0}, v.extents());

    T norm_sub = T{0};
    for (index_type i = k; i < m; ++i) {
      v(i) = R(i, k);
      norm_sub += v(i) * v(i);
    }
    norm_sub = std::sqrt(norm_sub);

    if (norm_sub < std::numeric_limits<T>::min()) {
      continue;
    }

    if (v(k) >= T{0}) {
      v(k) += norm_sub;
    } else {
      v(k) -= norm_sub;
    }

    T v_norm = T{0};
    for (index_type i = k; i < m; ++i) {
      v_norm += v(i) * v(i);
    }
    v_norm = std::sqrt(v_norm);
    for (index_type i = k; i < m; ++i) {
      v(i) /= v_norm;
    }

    // Apply reflection to R.
    for (index_type j = k; j < n; ++j) {
      T dot = T{0};
      for (index_type i = k; i < m; ++i) {
        dot += v(i) * R(i, j);
      }
      for (index_type i = k; i < m; ++i) {
        R(i, j) -= T{2} * v(i) * dot;
      }
    }

    // Apply reflection to Q.
    for (index_type i = 0; i < m; ++i) {
      T dot = T{0};
      for (index_type j = k; j < m; ++j) {
        dot += Q(i, j) * v(j);
      }
      for (index_type j = k; j < m; ++j) {
        Q(i, j) -= T{2} * dot * v(j);
      }
    }
  }

  return QRFullResult<T, M, N>{.Q = std::move(Q), .R = std::move(R)};
}

// ─────────────────────────────────────────────────────────────────────────────
// Gram-Schmidt QR (reduced)
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Reduced QR decomposition via classical Gram-Schmidt.
///
/// @param A  An m x n matrix with m >= n.
/// @return   A `QRReducedResult` with Q (m x n, orthonormal columns) and
///           R (n x n, upper triangular).
///
/// Q is computed by applying `orthogonalization::gram_schmidt()` to the
/// columns of A.  R is then computed as R = Q^T * A via element-level loops
/// (since zipper may not support general matrix-matrix multiplication).
///
/// This variant is simpler than Householder but less numerically stable.
/// For well-conditioned matrices the results are essentially identical.
template <concepts::Matrix Derived> auto qr_gram_schmidt(const Derived &A) {
  using AType = std::decay_t<Derived>;
  using T = typename AType::value_type;
  constexpr index_type M = AType::extents_type::static_extent(0);
  constexpr index_type N = AType::extents_type::static_extent(1);

  const index_type m = A.extent(0);
  const index_type n = A.extent(1);

  // Q = orthonormalised columns of A.
  Matrix<T, M, N> Q = utils::orthogonalization::gram_schmidt(A);

  // R = Q^T * A, computed element-by-element.
  // R(i, j) = sum_k Q(k, i) * A(k, j)   for i in [0, n), j in [0, n).
  Matrix<T, N, N> R(n, n);
  R = expression::nullary::Constant(T{0}, R.extents());

  for (index_type i = 0; i < n; ++i) {
    for (index_type j = 0; j < n; ++j) {
      T sum = T{0};
      for (index_type k = 0; k < m; ++k) {
        sum += Q(k, i) * A(k, j);
      }
      R(i, j) = sum;
    }
  }

  return QRReducedResult<T, M, N>{.Q = std::move(Q), .R = std::move(R)};
}

} // namespace zipper::utils::decomposition

#endif
