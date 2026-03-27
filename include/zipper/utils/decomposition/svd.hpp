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
#include <cmath>
#include <limits>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/utils/extents/extent_arithmetic.hpp>

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
    DynMat W(m, n);
    for (index_type i = 0; i < m; ++i) {
        for (index_type j = 0; j < n; ++j) { W(i, j) = A(i, j); }
    }

    // V accumulates right rotations (n x n identity initially).
    DynMat V(n, n);
    V = expression::nullary::
        Identity<T, std::dynamic_extent, std::dynamic_extent>(V.extents());

    const T eps = std::numeric_limits<T>::epsilon();
    const index_type max_sweeps = 100;

    for (index_type sweep = 0; sweep < max_sweeps; ++sweep) {
        // Check convergence: is W^T W close to diagonal?
        // We check that all off-diagonal entries of W^T W are small
        // relative to the diagonal.
        T off_diag_norm_sq = T{0};
        T diag_norm_sq = T{0};

        for (index_type j = 0; j < n; ++j) {
            T col_norm_sq = T{0};
            for (index_type i = 0; i < m; ++i) {
                col_norm_sq += W(i, j) * W(i, j);
            }
            diag_norm_sq += col_norm_sq * col_norm_sq;
        }

        for (index_type j1 = 0; j1 < n; ++j1) {
            for (index_type j2 = j1 + 1; j2 < n; ++j2) {
                T dot = T{0};
                for (index_type i = 0; i < m; ++i) {
                    dot += W(i, j1) * W(i, j2);
                }
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
                T a = T{0}, b = T{0}, d = T{0};
                for (index_type i = 0; i < m; ++i) {
                    a += W(i, j1) * W(i, j1);
                    b += W(i, j2) * W(i, j2);
                    d += W(i, j1) * W(i, j2);
                }

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

                // Apply right rotation to W: columns j1 and j2.
                //   W(:,j1)' =  c * W(:,j1) + s * W(:,j2)
                //   W(:,j2)' = -s * W(:,j1) + c * W(:,j2)
                for (index_type i = 0; i < m; ++i) {
                    T w1 = W(i, j1);
                    T w2 = W(i, j2);
                    W(i, j1) = c * w1 + s * w2;
                    W(i, j2) = -s * w1 + c * w2;
                }

                // Accumulate into V: columns j1 and j2.
                for (index_type i = 0; i < n; ++i) {
                    T v1 = V(i, j1);
                    T v2 = V(i, j2);
                    V(i, j1) = c * v1 + s * v2;
                    V(i, j2) = -s * v1 + c * v2;
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
    Vector<T, std::dynamic_extent> all_sigmas(n);
    for (index_type j = 0; j < n; ++j) {
        T norm_sq = T{0};
        for (index_type i = 0; i < m; ++i) { norm_sq += W(i, j) * W(i, j); }
        all_sigmas(j) = std::sqrt(norm_sq);
    }

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
            for (index_type i = 0; i < m; ++i) {
                U_result(i, k) = W(i, col) / S_result(k);
            }
        } else {
            // Zero singular value — choose an arbitrary orthonormal vector.
            // For simplicity, use a column from the identity (adjusted below
            // via Gram-Schmidt if needed).
            for (index_type i = 0; i < m; ++i) { U_result(i, k) = T{0}; }
            if (k < m) { U_result(k, k) = T{1}; }
        }

        // Vt is p x n: row k of Vt = column perm[k] of V, transposed.
        for (index_type j = 0; j < n; ++j) { Vt_result(k, j) = V(j, col); }
    }

    // Orthonormalise U columns for zero singular values using modified
    // Gram-Schmidt against the existing columns.
    for (index_type k = 0; k < p; ++k) {
        if (S_result(k) > std::numeric_limits<T>::min()) { continue; }

        // Try basis vectors e_0, e_1, ..., e_{m-1} until we find one
        // that is linearly independent from the existing U columns.
        for (index_type trial = 0; trial < m; ++trial) {
            // Start with e_{trial}.
            for (index_type i = 0; i < m; ++i) { U_result(i, k) = T{0}; }
            U_result(trial, k) = T{1};

            // Orthogonalise against all previous columns.
            for (index_type j = 0; j < k; ++j) {
                T dot = T{0};
                for (index_type i = 0; i < m; ++i) {
                    dot += U_result(i, k) * U_result(i, j);
                }
                for (index_type i = 0; i < m; ++i) {
                    U_result(i, k) -= dot * U_result(i, j);
                }
            }

            // Normalise.
            T norm_sq = T{0};
            for (index_type i = 0; i < m; ++i) {
                norm_sq += U_result(i, k) * U_result(i, k);
            }
            T norm = std::sqrt(norm_sq);
            if (norm > T{1e-10}) {
                for (index_type i = 0; i < m; ++i) { U_result(i, k) /= norm; }
                break;
            }
        }
    }

    return SVDResult<T, M, N>{.U = std::move(U_result),
                              .S = std::move(S_result),
                              .Vt = std::move(Vt_result)};
}

} // namespace zipper::utils::decomposition

#endif
