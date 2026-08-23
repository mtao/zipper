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
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/nullary/StaticConstant.hpp>
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

    // Scaling keeps the Gram quantities in range without changing the Jacobi
    // rotations. Leave non-finite inputs untouched; this unchecked API retains
    // its existing behavior for them.
    T input_scale = T{0};
    bool finite_input = true;
    for (index_type i = 0; i < m; ++i) {
        for (index_type j = 0; j < n; ++j) {
            const T value = A(i, j);
            finite_input = finite_input && std::isfinite(value);
            input_scale = std::max(input_scale, std::abs(value));
        }
    }

    // Working copy of A.
    DynMat W(A);
    if (finite_input && input_scale > T{0}) { W = W / input_scale; }

    // V accumulates right rotations (n x n identity initially).
    DynMat V(expression::nullary::
                 Identity<T, std::dynamic_extent, std::dynamic_extent>(n, n));

    const T eps = std::numeric_limits<T>::epsilon();
    const index_type max_sweeps = 100;

    for (index_type sweep = 0; sweep < max_sweeps; ++sweep) {
        bool converged = true;

        // Sweep: apply Jacobi rotations to all pairs (j1, j2) with j1 < j2.
        for (index_type j1 = 0; j1 < n; ++j1) {
            for (index_type j2 = j1 + 1; j2 < n; ++j2) {
                const T norm1 = W.col(j1).norm();
                const T norm2 = W.col(j2).norm();
                if (norm1 == T{0} || norm2 == T{0}) { continue; }

                const T correlation =
                    (W.col(j1) / norm1).dot(W.col(j2) / norm2);
                if (std::abs(correlation) <= eps) { continue; }
                converged = false;

                // Form a uniformly scaled 2x2 Gram matrix. This preserves the
                // angle while avoiding a*b and the overflow-prone tau*tau
                // formulation of the Jacobi tangent.
                const T pair_scale = std::max(norm1, norm2);
                const T x = norm1 / pair_scale;
                const T y = norm2 / pair_scale;
                const T delta = x * x - y * y;
                const T twice_dot = T{2} * correlation * x * y;
                const T radius = std::hypot(delta, twice_dot);
                T t;
                if (delta == T{0}) {
                    t = std::copysign(T{1}, twice_dot);
                } else {
                    t = twice_dot /
                        (delta + std::copysign(radius, delta));
                }
                const T c = T{1} / std::hypot(T{1}, t);
                const T s = t * c;

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
        if (converged) { break; }
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
    auto all_sigmas = W.colwise().norm().eval();

    // Build a permutation array sorted by descending singular value.
    std::vector<index_type> perm(n);
    std::iota(perm.begin(), perm.end(), index_type{0});
    std::sort(perm.begin(), perm.end(), [&](index_type a, index_type b) {
        return all_sigmas(a) > all_sigmas(b);
    });
    perm.resize(p);

    // Gather the top p columns in singular-value order.
    auto sorted_W = W.col_slice(perm);
    Matrix<T, M, P> U_result(m, p);
    Vector<T, P> S_result(all_sigmas(perm));
    Matrix<T, P, N> Vt_result(V.col_slice(perm).transpose());
    for (index_type k = 0; k < p; ++k) {
        if (S_result(k) > T{0}) {
            U_result.col(k) = sorted_W.col(k) / S_result(k);
        } else {
            S_result(k) = T{0};

            Vector<T, M> best(m);
            T best_norm_squared = T{-1};
            for (index_type direction = 0; direction < m; ++direction) {
                Vector<T, M> residual(m);
                residual = expression::nullary::Zero<T, M>(residual.extents());
                residual(direction) = T{1};

                // Reorthogonalization keeps the completion stable when the
                // preceding singular vectors are only numerically orthogonal.
                for (index_type pass = 0; pass < 2; ++pass) {
                    for (index_type previous = 0; previous < k; ++previous) {
                        auto u = U_result.col(previous);
                        residual -= residual.dot(u) * u;
                    }
                }

                const T norm_squared = residual.template norm_powered<2>();
                if (norm_squared > best_norm_squared) {
                    best = residual;
                    best_norm_squared = norm_squared;
                }
            }
            U_result.col(k) = best / std::sqrt(best_norm_squared);
        }
    }
    if (finite_input && input_scale > T{0}) { S_result *= input_scale; }

    return SVDResult<T, M, N>{.U = std::move(U_result),
                              .S = std::move(S_result),
                              .Vt = std::move(Vt_result)};
}

} // namespace zipper::utils::decomposition

#endif
