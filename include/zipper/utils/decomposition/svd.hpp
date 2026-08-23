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
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/nullary/StaticConstant.hpp>
#include <zipper/utils/decomposition/detail/scalar_math.hpp>
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
/// The scalar type must be a field closed under square root; ordinary exact
/// rational types therefore require a promoted or approximate square-root type.
template <concepts::Matrix Derived>
    requires detail::orthogonal_decomposition_scalar<
        typename std::decay_t<Derived>::value_type>
auto svd(const Derived &A)
    -> SVDResult<typename std::decay_t<Derived>::value_type,
                 std::decay_t<Derived>::extents_type::static_extent(0),
                 std::decay_t<Derived>::extents_type::static_extent(1)> {
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

    const T eps = detail::scalar_math::epsilon<T>();
    const index_type max_sweeps = 100;

    const auto column_norm = [m](const DynMat &matrix, index_type column) {
        T norm{0};
        for (index_type row = 0; row < m; ++row) {
            norm = detail::scalar_math::hypotenuse(norm, matrix(row, column));
        }
        return norm;
    };

    const auto apply_rotation = [](DynMat &matrix,
                                   index_type j1,
                                   index_type j2,
                                   const T &c,
                                   const T &s) {
        for (index_type row = 0; row < matrix.extent(0); ++row) {
            const T first = matrix(row, j1);
            const T second = matrix(row, j2);
            const T scale = std::max(
                detail::scalar_math::absolute_value(first),
                detail::scalar_math::absolute_value(second));
            if (scale == T{0}) { continue; }

            const T scaled_first = first / scale;
            const T scaled_second = second / scale;
            matrix(row, j1) =
                (scaled_first * c + scaled_second * s) * scale;
            matrix(row, j2) =
                (-scaled_first * s + scaled_second * c) * scale;
        }
    };

    for (index_type sweep = 0; sweep < max_sweeps; ++sweep) {
        bool converged = true;

        // Sweep: apply Jacobi rotations to all pairs (j1, j2) with j1 < j2.
        for (index_type j1 = 0; j1 < n; ++j1) {
            for (index_type j2 = j1 + 1; j2 < n; ++j2) {
                const T norm1 = column_norm(W, j1);
                const T norm2 = column_norm(W, j2);
                if (norm1 == T{0} || norm2 == T{0}) { continue; }

                T correlation{0};
                for (index_type row = 0; row < m; ++row) {
                    correlation = correlation +
                                  (W(row, j1) / norm1) *
                                      (W(row, j2) / norm2);
                }
                if (detail::scalar_math::absolute_value(correlation) <= eps) {
                    continue;
                }
                converged = false;

                // Form a uniformly scaled 2x2 Gram matrix. This preserves the
                // angle while avoiding a*b and the overflow-prone tau*tau
                // formulation of the Jacobi tangent.
                const T pair_scale = std::max(norm1, norm2);
                const T x = norm1 / pair_scale;
                const T y = norm2 / pair_scale;
                const T delta = x * x - y * y;
                const T twice_dot = T{2} * correlation * x * y;
                const T radius =
                    detail::scalar_math::hypotenuse(delta, twice_dot);
                T t{0};
                if (delta == T{0}) {
                    t = detail::scalar_math::copy_sign(T{1}, twice_dot);
                } else {
                    t = twice_dot /
                        (delta + detail::scalar_math::copy_sign(radius, delta));
                }
                const T c = T{1} / detail::scalar_math::hypotenuse(T{1}, t);
                const T s = t * c;

                apply_rotation(W, j1, j2, c, s);
                apply_rotation(V, j1, j2, c, s);
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
    Vector<T, N> all_sigmas(n);
    for (index_type column = 0; column < n; ++column) {
        all_sigmas(column) = column_norm(W, column);
    }

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
    for (index_type k = 0; k < p; ++k) {
        if (all_sigmas(perm[k]) > T{0}) {
            for (index_type row = 0; row < m; ++row) {
                U_result(row, k) =
                    sorted_W(row, k) / all_sigmas(perm[k]);
            }
        } else {
            all_sigmas(perm[k]) = T{0};

            Vector<T, M> best(m);
            T best_norm{0};
            bool has_best = false;
            for (index_type direction = 0; direction < m; ++direction) {
                Vector<T, M> residual(m);
                residual =
                    expression::nullary::Zero<T, M>(residual.extents());
                residual(direction) = T{1};

                // Reorthogonalization keeps the completion stable when the
                // preceding singular vectors are only numerically orthogonal.
                for (index_type pass = 0; pass < 2; ++pass) {
                    for (index_type previous = 0; previous < k; ++previous) {
                        auto u = U_result.col(previous);
                        T projection{0};
                        for (index_type row = 0; row < m; ++row) {
                            projection =
                                projection + residual(row) * u(row);
                        }
                        for (index_type row = 0; row < m; ++row) {
                            residual(row) =
                                residual(row) - projection * u(row);
                        }
                    }
                }

                T residual_norm{0};
                for (index_type row = 0; row < m; ++row) {
                    residual_norm = detail::scalar_math::hypotenuse(
                        residual_norm, residual(row));
                }
                if (!has_best || residual_norm > best_norm) {
                    best = residual;
                    best_norm = residual_norm;
                    has_best = true;
                }
            }
            for (index_type row = 0; row < m; ++row) {
                U_result(row, k) = best(row) / best_norm;
            }
        }
    }
    Vector<T, P> S_result(p);
    for (index_type k = 0; k < p; ++k) {
        S_result(k) = all_sigmas(perm[k]);
    }
    Matrix<T, P, N> Vt_result(V.col_slice(perm).transpose());

    return SVDResult<T, M, N>{.U = std::move(U_result),
                              .S = std::move(S_result),
                              .Vt = std::move(Vt_result)};
}

} // namespace zipper::utils::decomposition

#endif
