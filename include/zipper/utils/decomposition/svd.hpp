/// @file svd.hpp
/// @brief Singular Value Decomposition (SVD) via one-sided Jacobi rotations.
/// @ingroup decompositions
///
/// Given an m x n matrix A, computes:
///
///   A = U * diag(S) * V^T
///
/// where U (m x p) has orthonormal columns, S (p) contains the singular values
/// in descending order, and V^T (p x n) has orthonormal rows, with
/// p = min(m, n).
///
/// The algorithm uses one-sided Jacobi SVD: starting from A, we repeatedly
/// apply 2x2 Jacobi rotations on the right to diagonalise A^T A.  When
/// A^T A is diagonal, the columns of A * V are the left singular vectors
/// (up to normalisation) and the diagonal entries of A^T A are the squared
/// singular values.
///
/// This approach is simple, numerically robust, and well-suited for the
/// small matrices typical in FEM applications (2x2, 3x3 deformation
/// gradients).
///
/// SVD always succeeds (no rank requirement), so the function returns the
/// result struct directly rather than wrapping in `std::expected`.

#if !defined(ZIPPER_UTILS_DECOMPOSITION_SVD_HPP)
#define ZIPPER_UTILS_DECOMPOSITION_SVD_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/nullary/Unit.hpp>
#include <zipper/utils/extents/extent_arithmetic.hpp>
#include <zipper/utils/orthogonalization/gram_schmidt.hpp>

namespace zipper::utils::decomposition {

// ─────────────────────────────────────────────────────────────────────────────
// Result type
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a Singular Value Decomposition.
///
/// Given an m x n matrix A, produces:
///   A = U * diag(S) * Vt
///
/// where U (m x p) has orthonormal columns, S (p) contains the singular values
/// in descending order (non-negative), and Vt (p x n) has orthonormal rows,
/// with p = min(m, n).
template <typename T, index_type M, index_type N>
struct SVDResult {
    /// Scalar type of the decomposition.
    using value_type = T;

    static constexpr index_type P = extents::min(M, N);

    /// Left singular vectors U (m x p).
    Matrix<T, M, P> U;
    /// Singular values S (p), in descending order.
    Vector<T, P> S;
    /// Right singular vectors transposed Vt (p x n).
    Matrix<T, P, N> Vt;
};

// ─────────────────────────────────────────────────────────────────────────────
// SVD (one-sided Jacobi)
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Singular Value Decomposition via one-sided Jacobi rotations.
///
/// @param A  An m x n matrix.
/// @return   An `SVDResult` with U (m x p), S (p), Vt (p x n), where
///           p = min(m, n) and singular values are in descending order.
///
/// SVD always succeeds, so the result is returned directly (not wrapped in
/// `std::expected`).
template <concepts::Matrix Derived>
auto svd(const Derived &A) {
    using AType = std::decay_t<Derived>;
    using T = typename AType::value_type;
    constexpr index_type M = AType::extents_type::static_extent(0);
    constexpr index_type N = AType::extents_type::static_extent(1);
    constexpr index_type P = extents::min(M, N);

    const index_type m = A.extent(0);
    const index_type n = A.extent(1);
    const index_type p = std::min(m, n);

    // Handle degenerate cases.
    if (m == 0 || n == 0) {
        Matrix<T, M, P> U(m, p);
        Vector<T, P> S(p);
        Matrix<T, P, N> Vt(p, n);
        return SVDResult<T, M, N>{
            .U = std::move(U), .S = std::move(S), .Vt = std::move(Vt)};
    }

    // ──────────────────────────────────────────────────────────────────────
    // One-sided Jacobi SVD
    // ──────────────────────────────────────────────────────────────────────
    //
    // We want A = U * diag(S) * V^T.
    //
    // Strategy: work with W = A (m x n) and accumulate right rotations V.
    //   - Apply Jacobi rotations on the right: W := W * G(i,j,theta)
    //   - The rotation angle theta is chosen to zero out (W^T W)(i,j),
    //     i.e., to make columns i and j of W orthogonal.
    //   - When all off-diagonal entries of W^T W are zero, the columns of
    //     W are the left singular vectors (up to sign and normalisation)
    //     scaled by the singular values, and V accumulates all the rotations.
    //
    // After convergence:
    //   - S(k) = ||W(:,k)||
    //   - U(:,k) = W(:,k) / S(k)
    //   - V^T = V^T (accumulated)

    using DynMat = Matrix<T, std::dynamic_extent, std::dynamic_extent>;

    // Working copy of A.
    DynMat W(A);

    // V accumulates right rotations (n x n identity initially).
    DynMat V(expression::nullary::
                 Identity<T, std::dynamic_extent, std::dynamic_extent>(n, n));

    const T eps = std::numeric_limits<T>::epsilon();
    const index_type max_sweeps = 100;

    for (index_type sweep = 0; sweep < max_sweeps; ++sweep) {
        // Check convergence: is W^T W close to diagonal?
        // We check that all off-diagonal entries of W^T W are small
        // relative to the diagonal.

        // diag_norm_sq = sum of ||col_j||^4 = sum of (||col_j||^2)^2
        auto col_norms_sq = W.colwise().template norm_powered<2>();
        T diag_norm_sq = zipper::as_array(col_norms_sq).pow(T{2}).sum();

        T off_diag_norm_sq = T{0};
        for (index_type j1 = 0; j1 < n; ++j1) {
            for (index_type j2 = j1 + 1; j2 < n; ++j2) {
                T dot = W.col(j1).dot(W.col(j2));
                off_diag_norm_sq += T{2} * dot * dot;
            }
        }

        if (off_diag_norm_sq <= eps * eps * diag_norm_sq) { break; }

        // Sweep: apply Jacobi rotations to all pairs (j1, j2) with j1 < j2.
        for (index_type j1 = 0; j1 < n; ++j1) {
            for (index_type j2 = j1 + 1; j2 < n; ++j2) {
                // Compute the 2x2 Gram matrix for columns j1 and j2:
                //   G = [ a  d ]   where a = W(:,j1)^T W(:,j1)
                //       [ d  b ]         b = W(:,j2)^T W(:,j2)
                //                        d = W(:,j1)^T W(:,j2)
                T a = W.col(j1).dot(W.col(j1));
                T b = W.col(j2).dot(W.col(j2));
                T d = W.col(j1).dot(W.col(j2));

                // If d is negligible, columns are already orthogonal.
                if (std::abs(d) <= eps * std::sqrt(a * b)) { continue; }

                // Jacobi rotation angle to zero out d:
                //   tan(2*theta) = 2*d / (a - b)
                T c, s;
                if (std::abs(a - b) < eps * (a + b)) {
                    // a ≈ b → theta = pi/4
                    c = std::sqrt(T{0.5});
                    s = (d >= T{0}) ? c : -c;
                } else {
                    T tau = (a - b) / (T{2} * d);
                    T t;
                    if (tau >= T{0}) {
                        t = T{1} / (tau + std::sqrt(T{1} + tau * tau));
                    } else {
                        t = T{-1} / (-tau + std::sqrt(T{1} + tau * tau));
                    }
                    c = T{1} / std::sqrt(T{1} + t * t);
                    s = t * c;
                }

                // Build the 2x2 Givens rotation matrix:
                //   G = [ c  -s ]
                //       [ s   c ]
                Matrix<T, 2, 2> G{{{c, -s}, {s, c}}};

                // Apply right rotation to W: W(:,[j1,j2]) *= G
                {
                    auto W_sub = W.col_slice(std::array<index_type, 2>{j1, j2});
                    W_sub = (W_sub * G).eval();
                }

                // Accumulate into V: V(:,[j1,j2]) *= G
                {
                    auto V_sub = V.col_slice(std::array<index_type, 2>{j1, j2});
                    V_sub = (V_sub * G).eval();
                }
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Extract U, S, Vt from W and V
    // ──────────────────────────────────────────────────────────────────────
    // After convergence, W = A * V, and W's columns are orthogonal.
    //   S(k) = ||W(:,k)||
    //   U(:,k) = W(:,k) / S(k)
    //   Vt = V^T

    // Compute singular values and build index array for sorting.
    // We need to select the top p singular values (for wide matrices,
    // there are n columns but only p = min(m,n) non-trivial singular values).

    // First compute all n column norms.
    Vector<T, std::dynamic_extent> all_sigmas(W.colwise().norm());

    // Build a permutation array sorted by descending singular value.
    std::vector<index_type> perm(n);
    std::iota(perm.begin(), perm.end(), index_type{0});
    std::sort(perm.begin(), perm.end(), [&](index_type a, index_type b) {
        return all_sigmas(a) > all_sigmas(b);
    });

    // Extract the top p results.
    Matrix<T, M, P> U_result(m, p);
    Vector<T, P> S_result(p);
    Matrix<T, P, N> Vt_result(p, n);

    for (index_type k = 0; k < p; ++k) {
        index_type col = perm[k];
        S_result(k) = all_sigmas(col);

        if (S_result(k) > std::numeric_limits<T>::min()) {
            U_result.col(k) = W.col(col) / S_result(k);
        } else {
            // Zero singular value — seed with e_k so that Gram-Schmidt
            // below can orthonormalise it against the other columns.
            U_result.col(k) =
                expression::nullary::unit_vector<T>(m, k < m ? k : 0);
        }

        // Vt is p x n: row k of Vt = column perm[k] of V, transposed.
        Vt_result.row(k) = V.col(col);
    }

    // Orthonormalise U columns.  The non-zero-SV columns are already
    // orthonormal; Gram-Schmidt will leave them unchanged and make the
    // zero-SV seed columns orthogonal to everything else.
    orthogonalization::gram_schmidt_in_place(U_result);

    return SVDResult<T, M, N>{.U = std::move(U_result),
                              .S = std::move(S_result),
                              .Vt = std::move(Vt_result)};
}

} // namespace zipper::utils::decomposition

#endif
