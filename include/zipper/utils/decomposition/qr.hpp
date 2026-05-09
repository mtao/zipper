/// @file qr.hpp
/// @brief QR decomposition via Householder reflections and Gram-Schmidt.
/// @ingroup decompositions
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
///      matrix multiplication.  Requires m >= n.  This variant is simpler
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
///
/// @see zipper::expression::unary::TriangularView::solve — used by `qr_solve`
///      and `qr_solve_full` for back-substitution on the upper-triangular R.
/// @see zipper::expression::unary::TriangularView — the expression type used
///      to wrap R for triangular solve.
/// @see zipper::utils::inverse — uses `qr()` + `TriangularView::solve()` to
///      compute the general matrix inverse.
/// @see zipper::utils::orthogonalization::gram_schmidt — used by the
///      `qr_gram_schmidt` variant to orthonormalise columns.
/// @see zipper::utils::solver::SolverError — error type returned when R has a
///      zero pivot (matrix is rank-deficient).

#if !defined(ZIPPER_UTILS_DECOMPOSITION_QR_HPP)
#define ZIPPER_UTILS_DECOMPOSITION_QR_HPP

#include <algorithm>
#include <cmath>
#include <expected>
#include <limits>
#include <numeric>

#include <zipper/DataArray.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/unary/TriangularView.hpp>
#include <zipper/utils/decomposition/detail/householder.hpp>
#include <zipper/utils/extents/extent_arithmetic.hpp>
#include <zipper/utils/orthogonalization/gram_schmidt.hpp>
#include <zipper/utils/solver/result.hpp>

namespace zipper::utils::decomposition {

// ─────────────────────────────────────────────────────────────────────────────
// Result types
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a reduced (thin) QR decomposition.
///
/// Q is m x p and R is p x n, where p = min(m, n).
///
/// Calling `.solve(b)` computes x by forming c = Q^T * b and then solving
/// R * x = c via back substitution.  For square systems this is the exact
/// solution; for overdetermined systems (m > n) this is the least-squares
/// solution.
template <typename T, index_type M, index_type N>
struct QRReducedResult {
    /// Scalar type of the decomposition.
    using value_type = T;

    static constexpr index_type P = extents::min(M, N);

    /// Orthonormal matrix Q (m x p).
    Matrix<T, M, P> Q;
    /// Upper triangular matrix R (p x n).
    Matrix<T, P, N> R;

    /// @brief Solve A*x = b (or least-squares min||Ax-b||) using the stored
    ///        QR factors.
    ///
    /// Computes c = Q^T * b, then solves the leading p x p block of R against c
    /// via upper-triangular back substitution.
    ///
    /// @param b  Right-hand side vector of length m.
    /// @return   `std::expected<Vector<T,P>, SolverError>` — the solution on
    ///           success, or a breakdown error if R has a zero pivot.
    template <concepts::Vector BDerived>
    auto solve(const BDerived &b) const
        -> std::expected<Vector<T, P>, solver::SolverError> {
        using ResultVec = Vector<T, P>;
        using Result = std::expected<ResultVec, solver::SolverError>;

        const index_type p = Q.extent(1);

        // 1. Compute c = Q^T * b (p-dimensional vector).
        //    c(i) = Q.col(i) . b
        ResultVec c(p);
        for (index_type i = 0; i < p; ++i) { c(i) = Q.col(i).dot(b); }

        // 2. Solve R * x = c via upper-triangular back substitution.
        //    Extract the leading p x p block of R.
        Matrix<T, P, P> R_sq(R.leftCols(p));

        auto R_upper =
            expression::triangular_view<expression::TriangularMode::Upper>(
                R_sq);
        auto solve_result = R_upper.solve(c);

        if (!solve_result) {
            return Result{std::unexpected(std::move(solve_result.error()))};
        }

        return Result{std::move(*solve_result)};
    }
};

/// Result of a full QR decomposition.
///
/// Q is m x m (orthogonal) and R is m x n (upper trapezoidal).
///
/// Calling `.solve(b)` computes c = Q^T * b (first p entries) and then
/// solves R_top * x = c via back substitution, where p = min(m, n).
template <typename T, index_type M, index_type N>
struct QRFullResult {
    /// Scalar type of the decomposition.
    using value_type = T;

    static constexpr index_type P = extents::min(M, N);

    /// Orthogonal matrix Q (m x m).
    Matrix<T, M, M> Q;
    /// Upper trapezoidal matrix R (m x n).
    Matrix<T, M, N> R;

    /// @brief Solve A*x = b (or least-squares min||Ax-b||) using the stored
    ///        full QR factors.
    ///
    /// Computes c = Q^T * b (first p entries), then solves the leading p x p
    /// block of R against c via upper-triangular back substitution.
    ///
    /// @param b  Right-hand side vector of length m.
    /// @return   `std::expected<Vector<T,P>, SolverError>` — the solution on
    ///           success, or a breakdown error if R has a zero pivot.
    template <concepts::Vector BDerived>
    auto solve(const BDerived &b) const
        -> std::expected<Vector<T, P>, solver::SolverError> {
        using ResultVec = Vector<T, P>;
        using Result = std::expected<ResultVec, solver::SolverError>;

        const index_type n = R.extent(1);
        const index_type p = std::min(Q.extent(0), n);

        // 1. Compute c = Q^T * b, but only the first p entries matter.
        //    c(i) = Q.col(i) . b
        ResultVec c(p);
        for (index_type i = 0; i < p; ++i) { c(i) = Q.col(i).dot(b); }

        // 2. Solve R_top * x = c, where R_top is the leading p x p block of R.
        Matrix<T, P, P> R_sq(R.topRows(p).leftCols(p));

        auto R_upper =
            expression::triangular_view<expression::TriangularMode::Upper>(
                R_sq);
        auto solve_result = R_upper.solve(c);

        if (!solve_result) {
            return Result{std::unexpected(std::move(solve_result.error()))};
        }

        return Result{std::move(*solve_result)};
    }
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
template <concepts::Matrix Derived>
auto qr(const Derived &A) {
    using AType = std::decay_t<Derived>;
    using T = typename AType::value_type;
    constexpr index_type M = AType::extents_type::static_extent(0);
    constexpr index_type N = AType::extents_type::static_extent(1);
    constexpr index_type P = extents::min(M, N);

    const index_type m = A.extent(0);
    const index_type n = A.extent(1);
    const index_type p = std::min(m, n);

    // Work on a copy of A that will be transformed into R.
    Matrix<T, M, N> R_work(A);

    // Q_full accumulates the Householder reflections.  Start as identity.
    Matrix<T, M, M> Q_full(m, m);
    Q_full = expression::nullary::Identity<T, M, M>(Q_full.extents());

    for (index_type k = 0; k < p; ++k) {
        const index_type len = m - k;

        // Extract the sub-column R_work(k:m-1, k) and compute its
        // Householder vector.
        Vector<T, dynamic_extent> sub_col(R_work.col(k).segment(k, len));
        auto hh = detail::householder_vector(sub_col);
        if (!hh) { continue; }

        // Apply H to R_work from the left: rows k..m-1, columns k..n-1.
        detail::apply_householder_left(R_work, hh->v, k, k, n);

        // Apply H to Q_full from the right: rows 0..m-1, columns k..m-1.
        detail::apply_householder_right(Q_full, hh->v, 0, m, k);
    }

    // Extract the reduced Q (first p columns) and R (top p rows).
    Matrix<T, M, P> Q(Q_full.leftCols(p));
    Matrix<T, P, N> R(R_work.topRows(p));

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
template <concepts::Matrix Derived>
auto qr_full(const Derived &A) {
    using AType = std::decay_t<Derived>;
    using T = typename AType::value_type;
    constexpr index_type M = AType::extents_type::static_extent(0);
    constexpr index_type N = AType::extents_type::static_extent(1);

    const index_type m = A.extent(0);
    const index_type n = A.extent(1);
    const index_type p = std::min(m, n);

    // Work copy of A → becomes R.
    Matrix<T, M, N> R(A);

    // Accumulate Q starting from identity.
    Matrix<T, M, M> Q(m, m);
    Q = expression::nullary::Identity<T, M, M>(Q.extents());

    for (index_type k = 0; k < p; ++k) {
        const index_type len = m - k;

        // Extract sub-column R(k:m-1, k) and compute Householder vector.
        Vector<T, dynamic_extent> sub_col(R.col(k).segment(k, len));
        auto hh = detail::householder_vector(sub_col);
        if (!hh) { continue; }

        // Apply H to R from the left: rows k..m-1, columns k..n-1.
        detail::apply_householder_left(R, hh->v, k, k, n);

        // Apply H to Q from the right: rows 0..m-1, columns k..m-1.
        detail::apply_householder_right(Q, hh->v, 0, m, k);
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
/// columns of A.  R is then computed as R = Q^T * A via matrix multiplication.
///
/// This variant is simpler than Householder but less numerically stable.
/// For well-conditioned matrices the results are essentially identical.
template <concepts::Matrix Derived>
auto qr_gram_schmidt(const Derived &A) {
    using AType = std::decay_t<Derived>;
    using T = typename AType::value_type;
    constexpr index_type M = AType::extents_type::static_extent(0);
    constexpr index_type N = AType::extents_type::static_extent(1);

    // Q = orthonormalised columns of A.
    Matrix<T, M, N> Q = utils::orthogonalization::gram_schmidt(A);

    // R = Q^T * A.
    Matrix<T, N, N> R(Q.transpose() * A);

    return QRReducedResult<T, M, N>{.Q = std::move(Q), .R = std::move(R)};
}

// ─────────────────────────────────────────────────────────────────────────────
// QR solve (reduced)
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Solve Ax = b via reduced QR decomposition.
///
/// Factors A = Q * R (reduced), then delegates to `QRReducedResult::solve(b)`.
///
/// For square systems (m == n) this computes the exact solution.
/// For overdetermined systems (m > n) this computes the least-squares
/// solution  min ||Ax - b||_2.
///
/// Requires m >= n (the matrix must be square or tall).
///
/// @param A  An m x n matrix with m >= n.
/// @param b  Right-hand side vector of length m.
/// @return   `std::expected<Vector<T, N>, SolverError>` — the solution on
///           success, or a breakdown error if R has a zero pivot.
template <concepts::Matrix ADerived, concepts::Vector BDerived>
auto qr_solve(const ADerived &A, const BDerived &b) {
    auto result = qr(A);
    return result.solve(b);
}

// ─────────────────────────────────────────────────────────────────────────────
// QR solve (full)
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Solve Ax = b via full QR decomposition.
///
/// Factors A = Q * R (full), then delegates to `QRFullResult::solve(b)`.
///
/// Uses the full Q (m x m) factorisation.  For square systems the result
/// is the same as `qr_solve`.  For overdetermined systems this also gives
/// the least-squares solution but using the full orthogonal Q, which may
/// be useful when Q itself is needed.
///
/// Requires m >= n.
///
/// @param A  An m x n matrix with m >= n.
/// @param b  Right-hand side vector of length m.
/// @return   `std::expected<Vector<T, N>, SolverError>` — the solution on
///           success, or a breakdown error if R has a zero pivot.
template <concepts::Matrix ADerived, concepts::Vector BDerived>
auto qr_solve_full(const ADerived &A, const BDerived &b) {
    auto result = qr_full(A);
    return result.solve(b);
}

// ─────────────────────────────────────────────────────────────────────────────
// Column-pivoted QR result type
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a column-pivoted QR decomposition.
///
/// Given an m x n matrix A, produces:
///   A * P = Q * R
/// equivalently:
///   A = Q * R * P^T
///
/// where Q (m x p, p = min(m,n)) has orthonormal columns, R (p x n) is upper
/// triangular, and P is a column permutation matrix.  Column pivoting selects,
/// at each step, the remaining column with the largest 2-norm, guaranteeing
/// that the diagonal entries of R are in non-increasing magnitude:
///   |R(0,0)| >= |R(1,1)| >= ... >= |R(p-1,p-1)|
///
/// This ordering makes the factorisation suitable for reliable numerical rank
/// determination: the rank is the number of leading diagonal entries that
/// exceed a tolerance.
template <typename T, index_type M, index_type N>
struct QRColPivotResult {
    /// Scalar type of the decomposition.
    using value_type = T;

    static constexpr index_type P = extents::min(M, N);

    /// Orthonormal matrix Q (m x p, p = min(m,n)).
    Matrix<T, M, P> Q;
    /// Upper triangular matrix R (p x n).
    Matrix<T, P, N> R;
    /// Column permutation: col_perm(j) = original column index at position j.
    DataArray<index_type, N> col_perm;

    /// @brief Compute the numerical rank from the R diagonal.
    ///
    /// Returns the number of leading diagonal entries of R whose absolute value
    /// exceeds `tol * |R(0,0)|`.  If no tolerance is provided, a default based
    /// on machine epsilon and the matrix dimensions is used.
    ///
    /// @param tol_override  Optional explicit tolerance multiplier.
    /// @return  The numerical rank.
    auto rank(T tol_override = T{-1}) const -> index_type {
        const index_type m = Q.extent(0);
        const index_type n = R.extent(1);
        const index_type p = std::min(m, n);

        if (p == 0) return 0;

        // Default tolerance: eps * max(m, n).
        T tol = tol_override;
        if (tol < T{0}) {
            tol = std::numeric_limits<T>::epsilon()
                  * static_cast<T>(std::max(m, n));
        }

        const T threshold = tol * std::abs(R(0, 0));

        index_type r = 0;
        for (index_type i = 0; i < p; ++i) {
            if (std::abs(R(i, i)) > threshold)
                ++r;
            else
                break; // Diagonal is non-increasing, so all subsequent are <=
                       // this.
        }
        return r;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Column-pivoted Householder QR
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Column-pivoted QR decomposition via Householder reflections.
///
/// At each step k, the algorithm selects the remaining column (index >= k) with
/// the largest 2-norm and swaps it into position k before applying the
/// Householder reflection.  This guarantees |R(k,k)| is non-increasing.
///
/// @param A  An m x n matrix.
/// @return   A `QRColPivotResult` with Q (m x p), R (p x n), and col_perm.
template <concepts::Matrix Derived>
auto qr_col_pivot(const Derived &A) {
    using AType = std::decay_t<Derived>;
    using T = typename AType::value_type;
    constexpr index_type M = AType::extents_type::static_extent(0);
    constexpr index_type N = AType::extents_type::static_extent(1);
    constexpr index_type P = extents::min(M, N);

    const index_type m = A.extent(0);
    const index_type n = A.extent(1);
    const index_type p = std::min(m, n);

    // Work on a copy of A that will be transformed into R.
    Matrix<T, M, N> R_work(A);

    // Accumulate Q starting from identity (m x m), extract first p cols later.
    Matrix<T, M, M> Q_full(m, m);
    Q_full = expression::nullary::Identity<T, M, M>(Q_full.extents());

    // Column permutation (identity initially).
    DataArray<index_type, N> col_perm(n);
    std::iota(col_perm.begin(), col_perm.end(), index_type{0});

    // Precompute column norms squared (diagonal of the Gram matrix A^T A)
    // for efficient pivot selection via Businger-Golub norm downdating.
    Vector<T, N> col_norms_sq(R_work.colwise().template norm_powered<2>());

    for (index_type k = 0; k < p; ++k) {
        // ── Column pivoting: find the column (>= k) with largest remaining
        // norm.
        index_type max_col = k;
        T max_norm = col_norms_sq(k);
        for (index_type j = k + 1; j < n; ++j) {
            if (col_norms_sq(j) > max_norm) {
                max_norm = col_norms_sq(j);
                max_col = j;
            }
        }

        // Swap columns k and max_col in R_work, col_perm, and col_norms_sq.
        if (max_col != k) {
            std::swap(col_perm(k), col_perm(max_col));
            std::swap(col_norms_sq(k), col_norms_sq(max_col));
            for (index_type i = 0; i < m; ++i) {
                std::swap(R_work(i, k), R_work(i, max_col));
            }
        }

        // ── Householder reflection.
        const index_type len = m - k;
        Vector<T, dynamic_extent> sub_col(R_work.col(k).segment(k, len));
        auto hh = detail::householder_vector(sub_col);
        if (!hh) { continue; }

        // Apply H to R_work from the left.
        detail::apply_householder_left(R_work, hh->v, k, k, n);

        // Apply H to Q_full from the right.
        detail::apply_householder_right(Q_full, hh->v, 0, m, k);

        // ── Update column norms for remaining columns (downdate).
        // After the reflection, R_work(k, j) for j > k has changed.
        // The remaining norm squared of column j (rows k+1..m-1) is
        // col_norms_sq(j) - R_work(k, j)^2.  This avoids recomputing from
        // scratch and is the standard technique (Businger-Golub).
        for (index_type j = k + 1; j < n; ++j) {
            col_norms_sq(j) -= R_work(k, j) * R_work(k, j);
            // Guard against negative due to rounding.
            if (col_norms_sq(j) < T{0}) col_norms_sq(j) = T{0};
        }
    }

    // Extract the reduced Q (first p columns) and R (top p rows).
    Matrix<T, M, P> Q(Q_full.leftCols(p));
    Matrix<T, P, N> R(R_work.topRows(p));

    return QRColPivotResult<T, M, N>{
        .Q = std::move(Q), .R = std::move(R), .col_perm = std::move(col_perm)};
}

// ─────────────────────────────────────────────────────────────────────────────
// Rank via column-pivoted QR
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Compute the numerical rank of a matrix via column-pivoted QR.
///
/// This is a convenience function that performs a column-pivoted QR
/// decomposition and returns the rank.
///
/// @param A  An m x n matrix.
/// @return   The numerical rank.
template <concepts::Matrix Derived>
auto rank(const Derived &A) -> index_type {
    return qr_col_pivot(A).rank();
}

} // namespace zipper::utils::decomposition

#endif
