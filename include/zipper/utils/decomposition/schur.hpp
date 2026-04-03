/// @file schur.hpp
/// @brief Real Schur decomposition via Hessenberg reduction + QR iteration.
/// @ingroup decompositions
///
/// Given a square n x n real matrix A, computes the real Schur decomposition:
///
///   A = U * T * U^T
///
/// where U is orthogonal and T is quasi-upper-triangular (real Schur form).
/// The diagonal of T consists of 1x1 blocks (real eigenvalues) and 2x2 blocks
/// (complex conjugate eigenvalue pairs).
///
/// The algorithm proceeds in two phases:
///   1. **Hessenberg reduction**: transform A to upper Hessenberg form H
///      using Householder reflections, with A = Q * H * Q^T.
///   2. **QR iteration**: apply implicit QR steps to H until it converges to
///      quasi-triangular form T.
///
/// Two QR iteration strategies are available:
///   - **Francis double-shift** (default): uses implicit double-shift with
///     3x3 Householder reflectors. Handles complex conjugate eigenvalue pairs
///     naturally with real arithmetic. Standard LAPACK approach.
///   - **Wilkinson single-shift**: uses implicit single-shift with Givens
///     rotations. Simpler per iteration but may need more iterations. Falls
///     back to a22 shift for complex eigenvalues.
///
/// @code
///   auto A = Matrix<double, 3, 3>({{0, 1, 0}, {0, 0, 1}, {1, 0, 0}});
///   auto result = zipper::utils::decomposition::schur(A);
///   if (result) {
///       auto& [U, T_mat] = *result;
///       // A ≈ U * T_mat * U^T
///   }
///
///   // Explicitly select a strategy:
///   auto r2 = zipper::utils::decomposition::schur(
///       A, 0, SchurStrategy::wilkinson);
/// @endcode

#pragma once

#include <algorithm>
#include <cmath>
#include <expected>
#include <limits>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/utils/decomposition/detail/givens.hpp>
#include <zipper/utils/decomposition/detail/householder.hpp>
#include <zipper/utils/determinant.hpp>
#include <zipper/utils/solver/result.hpp>

namespace zipper::utils::decomposition {

// ─────────────────────────────────────────────────────────────────────────────
// Strategy enum
// ─────────────────────────────────────────────────────────────────────────────

/// Selects the QR iteration strategy for the Schur decomposition.
enum class SchurStrategy {
    /// Francis implicit double-shift QR (default).
    /// Uses 3x3 Householder reflectors; handles complex conjugate eigenvalue
    /// pairs with purely real arithmetic.  Standard LAPACK algorithm.
    francis,
    /// Wilkinson implicit single-shift QR.
    /// Uses Givens rotations; simpler per step, may need more iterations.
    /// Falls back to a22 shift for complex eigenvalue pairs.
    wilkinson,
};

// ─────────────────────────────────────────────────────────────────────────────
// Result type
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a real Schur decomposition.
///
/// A = U * T_mat * U^T, where U is orthogonal and T_mat is quasi-upper-
/// triangular (real Schur form).
template <typename T, index_type N = dynamic_extent>
struct SchurResult {
    /// Orthogonal matrix U.
    Matrix<T, N, N> U;
    /// Quasi-upper-triangular matrix T (real Schur form).
    Matrix<T, N, N> T_mat;
};

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace schur_detail {

    /// @brief Reduce a square matrix to upper Hessenberg form using Householder
    ///        reflections.
    ///
    /// On exit, H is upper Hessenberg (H(i,j) = 0 for i > j+1) and Q is
    /// orthogonal such that A = Q * H * Q^T.
    template <typename T>
    auto hessenberg_reduce(Matrix<T, dynamic_extent, dynamic_extent> &H,
                           Matrix<T, dynamic_extent, dynamic_extent> &Q)
        -> void {
        const index_type n = H.extent(0);

        Q = expression::nullary::Identity<T, dynamic_extent, dynamic_extent>(n,
                                                                             n);

        for (index_type k = 0; k < n - 2; ++k) {
            const index_type m = n - k - 1; // length of the sub-column

            // Extract the sub-column H(k+1:n, k) into an owning vector.
            Vector<T, dynamic_extent> v(H.col(k).segment(k + 1, m));

            // Compute the Householder vector.
            auto hh = detail::householder_vector(v);
            if (!hh) { continue; }

            // H <- P H P  where P = I - 2 v v^T acts on rows k+1:n.
            detail::apply_householder_left(H, hh->v, k + 1, k, n);
            detail::apply_householder_right(H, hh->v, 0, n, k + 1);

            // Accumulate: Q <- Q P
            detail::apply_householder_right(Q, hh->v, 0, n, k + 1);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Francis double-shift QR iteration (Phase 2)
    // ─────────────────────────────────────────────────────────────────────────────

    /// @brief Perform Francis implicit double-shift QR iteration on a
    /// Hessenberg
    ///        matrix H, accumulating rotations into U_acc.
    ///
    /// Uses 3x3 Householder reflectors for bulge chasing.  The two shifts are
    /// the eigenvalues of the trailing 2x2 block; their sum and product are
    /// always real, so no complex arithmetic is required.
    template <typename T>
    auto francis_qr_iteration(Matrix<T, dynamic_extent, dynamic_extent> &H,
                              Matrix<T, dynamic_extent, dynamic_extent> &U_acc,
                              index_type max_iter)
        -> std::expected<void, solver::SolverError> {
        const index_type n = H.extent(0);
        const T eps = std::numeric_limits<T>::epsilon();
        index_type p = n; // active sub-matrix is H[0:p, 0:p]
        index_type total_iter = 0;

        while (p > 2) {
            if (total_iter >= max_iter) {
                return std::unexpected(solver::SolverError{
                    .kind = solver::SolverError::Kind::diverged,
                    .message = "schur: Francis QR iteration did not converge"});
            }

            // Check for deflation at the bottom (1x1 block).
            T threshold =
                eps * (std::abs(H(p - 2, p - 2)) + std::abs(H(p - 1, p - 1)));
            if (threshold == T{0}) { threshold = eps; }

            if (std::abs(H(p - 1, p - 2)) <= threshold) {
                H(p - 1, p - 2) = T{0};
                --p;
                continue;
            }

            // Check for deflation of a 2x2 block at the bottom.
            if (p >= 3) {
                T thresh2 =
                    eps
                    * (std::abs(H(p - 3, p - 3)) + std::abs(H(p - 2, p - 2)));
                if (thresh2 == T{0}) { thresh2 = eps; }
                if (std::abs(H(p - 2, p - 3)) <= thresh2) {
                    H(p - 2, p - 3) = T{0};
                    p -= 2;
                    continue;
                }
            }

            // Find the top of the unreduced Hessenberg block.
            index_type q = p - 1;
            while (q > 0) {
                T thresh_q =
                    eps * (std::abs(H(q - 1, q - 1)) + std::abs(H(q, q)));
                if (thresh_q == T{0}) { thresh_q = eps; }
                if (std::abs(H(q, q - 1)) <= thresh_q) {
                    H(q, q - 1) = T{0};
                    break;
                }
                --q;
            }

            // If the unreduced block is already 1x1 or 2x2, deflate directly.
            if (p - q <= 2) {
                p = q;
                continue;
            }

            // --- Francis double shift ---
            // Shifts are the eigenvalues of the trailing 2x2 block H(p-2:p,
            // p-2:p). We only need their sum (s) and product (t) which are
            // always real.
            auto trailing = H.slice(zipper::slice(p - 2, index_type{2}),
                                    zipper::slice(p - 2, index_type{2}));
            T s = trailing.trace();
            T t = utils::determinant(trailing);

            // First column of M = H^2 - s*H + t*I (only first 3 entries are
            // non-zero because H is upper Hessenberg).
            auto leading = H.slice(zipper::slice(q, index_type{2}),
                                   zipper::slice(q, index_type{2}));
            T h10 = leading(1, 0);
            T h21 = (q + 2 < n) ? H(q + 2, q + 1) : T{0};

            T x = leading(0, 0) * leading(0, 0) + leading(0, 1) * h10
                  - s * leading(0, 0) + t;
            T y = h10 * (leading.trace() - s);
            T z = h10 * h21;

            // Chase the bulge from q to p-3 using 3x3 Householder reflectors,
            // then finish with a 2x2 Givens rotation.
            for (index_type k = q; k <= p - 3; ++k) {
                // Build Householder reflector for [x, y, z].
                index_type r = std::min(index_type{3}, p - k);

                Vector<T, dynamic_extent> v(r);
                v(0) = x;
                if (r > 1) v(1) = y;
                if (r > 2) v(2) = z;

                auto hh = detail::householder_vector(v);
                if (!hh) {
                    // Bulge has vanished; skip this reflector.
                    if (k + 3 < p) {
                        x = H(k + 1, k);
                        y = H(k + 2, k);
                        z = (k + 3 < p) ? H(k + 3, k) : T{0};
                    }
                    continue;
                }

                // Determine row/col bounds for applying the reflector.
                index_type r0 =
                    (k > q) ? (k - 1) : q; // leftmost column affected
                index_type c1 = n; // right boundary

                // Apply from the left: H(k:k+r, r0:c1)
                detail::apply_householder_left(H, hh->v, k, r0, c1);
                // Apply from the right: H(0:min(k+r+1,p), k:k+r)
                index_type r1 = std::min(k + r + 1, p);
                detail::apply_householder_right(H, hh->v, 0, r1, k);
                // Accumulate into U: U(:, k:k+r)
                detail::apply_householder_right(U_acc, hh->v, 0, n, k);

                // Set up the bulge entries for the next iteration.
                if (k + 3 < p) {
                    x = H(k + 1, k);
                    y = H(k + 2, k);
                    z = (k + 3 < p) ? H(k + 3, k) : T{0};
                }
            }

            // Final 2x2 Givens rotation for the last bulge entry at (p-2, p-3).
            if (p >= 3 && p - 2 > q) {
                auto [c, s_val] =
                    detail::givens_params(H(p - 2, p - 3), H(p - 1, p - 3));
                detail::apply_givens_rows(H, p - 2, p - 1, c, s_val, 0, n);
                detail::apply_givens_cols(H, p - 2, p - 1, c, s_val, 0, n);
                detail::apply_givens_cols(U_acc, p - 2, p - 1, c, s_val, 0, n);
            }

            // Clean up numerical noise below the sub-diagonal.
            for (index_type i = q + 2; i < p; ++i) {
                for (index_type j = q; j < i - 1; ++j) { H(i, j) = T{0}; }
            }

            ++total_iter;
        }

        return {};
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Wilkinson single-shift QR iteration (Phase 2)
    // ─────────────────────────────────────────────────────────────────────────────

    /// @brief Perform Wilkinson implicit single-shift QR iteration on a
    ///        Hessenberg matrix H, accumulating rotations into U_acc.
    ///
    /// Uses Givens rotations.  The shift is the eigenvalue of the trailing 2x2
    /// block closest to H(p-1, p-1).  For complex conjugate pairs the shift
    /// falls back to a22, which is correct for real Schur form (the 2x2 block
    /// will not deflate further).
    template <typename T>
    auto
        wilkinson_qr_iteration(Matrix<T, dynamic_extent, dynamic_extent> &H,
                               Matrix<T, dynamic_extent, dynamic_extent> &U_acc,
                               index_type max_iter)
            -> std::expected<void, solver::SolverError> {
        const index_type n = H.extent(0);
        const T eps = std::numeric_limits<T>::epsilon();
        index_type p = n; // active sub-matrix is H[0:p, 0:p]
        index_type total_iter = 0;

        while (p > 2) {
            if (total_iter >= max_iter) {
                return std::unexpected(solver::SolverError{
                    .kind = solver::SolverError::Kind::diverged,
                    .message =
                        "schur: Wilkinson QR iteration did not converge"});
            }

            // Check for deflation at the bottom.
            T threshold =
                eps * (std::abs(H(p - 2, p - 2)) + std::abs(H(p - 1, p - 1)));
            if (threshold == T{0}) { threshold = eps; }

            if (std::abs(H(p - 1, p - 2)) <= threshold) {
                H(p - 1, p - 2) = T{0};
                --p;
                continue;
            }

            // Check if we have a 2x2 block at the bottom with complex
            // eigenvalues. The 2x2 block H(p-2:p, p-2:p) has complex
            // eigenvalues iff discriminant < 0. In that case, check if H(p-2,
            // p-3) is negligible to deflate the whole 2x2 block.
            if (p >= 3) {
                T thresh2 =
                    eps
                    * (std::abs(H(p - 3, p - 3)) + std::abs(H(p - 2, p - 2)));
                if (thresh2 == T{0}) { thresh2 = eps; }
                if (std::abs(H(p - 2, p - 3)) <= thresh2) {
                    H(p - 2, p - 3) = T{0};
                    p -= 2;
                    continue;
                }
            }

            // Find the top of the unreduced Hessenberg block.
            index_type q = p - 1;
            while (q > 0) {
                T thresh_q =
                    eps * (std::abs(H(q - 1, q - 1)) + std::abs(H(q, q)));
                if (thresh_q == T{0}) { thresh_q = eps; }
                if (std::abs(H(q, q - 1)) <= thresh_q) {
                    H(q, q - 1) = T{0};
                    break;
                }
                --q;
            }

            // If the block is already 1x1 or 2x2, deflate directly.
            if (p - q <= 2) {
                p = q;
                continue;
            }

            // Wilkinson shift: eigenvalue of the bottom-right 2x2 block
            // closest to H(p-1, p-1).
            auto trailing = H.slice(zipper::slice(p - 2, index_type{2}),
                                    zipper::slice(p - 2, index_type{2}));
            T a22 = trailing(1, 1);

            T trace = trailing.trace();
            T det = utils::determinant(trailing);
            T disc = trace * trace - T{4} * det;

            T shift;
            if (disc >= T{0}) {
                // Two real eigenvalues; pick the one closer to a22.
                T sqrt_disc = std::sqrt(disc);
                T e1 = (trace + sqrt_disc) / T{2};
                T e2 = (trace - sqrt_disc) / T{2};
                shift = (std::abs(e1 - a22) < std::abs(e2 - a22)) ? e1 : e2;
            } else {
                // Complex conjugate pair; use a22 as the shift (exceptional
                // shift).
                shift = a22;
            }

            // Implicit QR step with Givens rotations on H[q:p, q:p].
            // First rotation: zero out (H(q,q) - shift, H(q+1,q)).
            T x = H(q, q) - shift;
            T y = H(q + 1, q);

            for (index_type k = q; k < p - 1; ++k) {
                auto [c, sv] = detail::givens_params(x, y);

                // Apply G from the left to rows k and k+1.
                detail::apply_givens_rows(H, k, k + 1, c, sv, 0, n);
                // Apply G^T from the right to columns k and k+1.
                detail::apply_givens_cols(H, k, k + 1, c, sv, 0, n);
                // Accumulate into U.
                detail::apply_givens_cols(U_acc, k, k + 1, c, sv, 0, n);

                // Prepare for next iteration: the bulge entry.
                if (k + 2 < p) {
                    x = H(k + 1, k);
                    y = H(k + 2, k);
                }
            }

            // Clean up sub-sub-diagonal entries created by numerical noise.
            for (index_type i = q + 2; i < p; ++i) {
                for (index_type j = q; j < i - 1; ++j) { H(i, j) = T{0}; }
            }

            ++total_iter;
        }

        return {};
    }

} // namespace schur_detail

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Compute the real Schur decomposition of a square matrix.
///
/// Given A (n x n), computes A = U * T * U^T where U is orthogonal and T is
/// quasi-upper-triangular (real Schur form).
///
/// @param A         A square matrix satisfying `concepts::Matrix`.
/// @param max_iter  Maximum number of QR iterations (default: 200*n).
/// @param strategy  QR iteration strategy (default: Francis double-shift).
/// @return          On success, `SchurResult{U, T_mat}`.
///                  On failure (did not converge), a `SolverError`.
template <concepts::Matrix Derived>
auto schur(const Derived &A,
           index_type max_iter = 0,
           SchurStrategy strategy = SchurStrategy::francis)
    -> std::expected<
        SchurResult<typename std::decay_t<Derived>::value_type, dynamic_extent>,
        solver::SolverError> {
    using AType = std::decay_t<Derived>;
    using T = typename AType::value_type;

    const index_type n = A.extent(0);

    if (n != A.extent(1)) {
        return std::unexpected(
            solver::SolverError{.kind = solver::SolverError::Kind::breakdown,
                                .message = "schur: matrix must be square"});
    }

    if (max_iter == 0) { max_iter = 200 * std::max(n, index_type{1}); }

    // Trivial cases.
    if (n == 0) {
        Matrix<T, dynamic_extent, dynamic_extent> U_r(0, 0);
        Matrix<T, dynamic_extent, dynamic_extent> T_r(0, 0);
        return SchurResult<T, dynamic_extent>{.U = std::move(U_r),
                                              .T_mat = std::move(T_r)};
    }
    if (n == 1) {
        Matrix<T, dynamic_extent, dynamic_extent> U_r(
            expression::nullary::Identity<T, dynamic_extent, dynamic_extent>(
                1, 1));
        Matrix<T, dynamic_extent, dynamic_extent> T_r(1, 1);
        T_r(0, 0) = A(0, 0);
        return SchurResult<T, dynamic_extent>{.U = std::move(U_r),
                                              .T_mat = std::move(T_r)};
    }

    // Phase 1: Hessenberg reduction.
    Matrix<T, dynamic_extent, dynamic_extent> H(A);
    Matrix<T, dynamic_extent, dynamic_extent> U_acc(n, n);
    schur_detail::hessenberg_reduce(H, U_acc);

    // Clean up numerical noise below the sub-diagonal.
    for (index_type i = 2; i < n; ++i) {
        for (index_type j = 0; j < i - 1; ++j) { H(i, j) = T{0}; }
    }

    // Phase 2: QR iteration (strategy-dependent).
    std::expected<void, solver::SolverError> qr_result;
    switch (strategy) {
    case SchurStrategy::francis:
        qr_result = schur_detail::francis_qr_iteration(H, U_acc, max_iter);
        break;
    case SchurStrategy::wilkinson:
        qr_result = schur_detail::wilkinson_qr_iteration(H, U_acc, max_iter);
        break;
    }
    if (!qr_result) { return std::unexpected(qr_result.error()); }

    // Handle the remaining 2x2 block (p <= 2): already in Schur form.

    return SchurResult<T, dynamic_extent>{.U = std::move(U_acc),
                                          .T_mat = std::move(H)};
}

} // namespace zipper::utils::decomposition
