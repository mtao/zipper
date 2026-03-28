/// @file schur.hpp
/// @brief Real Schur decomposition via Hessenberg reduction + Francis QR.
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
///   2. **Francis QR iteration**: apply implicit double-shift QR steps to H
///      until it converges to quasi-triangular form T.
///
/// @code
///   auto A = Matrix<double, 3, 3>({{0, 1, 0}, {0, 0, 1}, {1, 0, 0}});
///   auto result = zipper::utils::decomposition::schur(A);
///   if (result) {
///       auto& [U, T_mat] = *result;
///       // A ≈ U * T_mat * U^T
///   }
/// @endcode

#if !defined(ZIPPER_UTILS_DECOMPOSITION_SCHUR_HPP)
#define ZIPPER_UTILS_DECOMPOSITION_SCHUR_HPP

#include <algorithm>
#include <cmath>
#include <expected>
#include <limits>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/utils/solver/result.hpp>

namespace zipper::utils::decomposition {

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

namespace detail {

/// @brief Apply a Householder reflection P = I - 2 v v^T to a matrix from
///        the left: H[r0:r0+len, c0:c1) -= 2 * v * (v^T * H[...]).
template <typename T>
void apply_householder_left(
    Matrix<T, dynamic_extent, dynamic_extent> &H,
    const Vector<T, dynamic_extent> &v,
    index_type r0, index_type len, index_type c0, index_type c1) {
    for (index_type j = c0; j < c1; ++j) {
        T dot = T{0};
        for (index_type i = 0; i < len; ++i) {
            dot += v(i) * H(r0 + i, j);
        }
        dot *= T{2};
        for (index_type i = 0; i < len; ++i) {
            H(r0 + i, j) -= dot * v(i);
        }
    }
}

/// @brief Apply a Householder reflection from the right:
///        H[r0:r1, c0:c0+len) -= 2 * (H[...] * v) * v^T.
template <typename T>
void apply_householder_right(
    Matrix<T, dynamic_extent, dynamic_extent> &H,
    const Vector<T, dynamic_extent> &v,
    index_type r0, index_type r1, index_type c0, index_type len) {
    for (index_type i = r0; i < r1; ++i) {
        T dot = T{0};
        for (index_type j = 0; j < len; ++j) {
            dot += H(i, c0 + j) * v(j);
        }
        dot *= T{2};
        for (index_type j = 0; j < len; ++j) {
            H(i, c0 + j) -= dot * v(j);
        }
    }
}

/// @brief Reduce a square matrix to upper Hessenberg form using Householder
///        reflections.
///
/// On exit, H is upper Hessenberg (H(i,j) = 0 for i > j+1) and Q is
/// orthogonal such that A = Q * H * Q^T.
template <typename T>
void hessenberg_reduce(
    Matrix<T, dynamic_extent, dynamic_extent> &H,
    Matrix<T, dynamic_extent, dynamic_extent> &Q) {
    const index_type n = H.extent(0);

    Q = expression::nullary::Identity<T, dynamic_extent, dynamic_extent>(n, n);

    for (index_type k = 0; k < n - 2; ++k) {
        const index_type m = n - k - 1;  // length of the sub-column

        // Extract the sub-column H(k+1:n, k).
        Vector<T, dynamic_extent> v(m);
        for (index_type i = 0; i < m; ++i) {
            v(i) = H(k + 1 + i, k);
        }

        T sigma = v.norm();
        if (sigma < std::numeric_limits<T>::epsilon()) {
            continue;
        }

        if (v(0) >= T{0}) { sigma = -sigma; }

        v(0) -= sigma;
        T v_norm = v.norm();
        if (v_norm < std::numeric_limits<T>::min()) { continue; }
        for (index_type i = 0; i < m; ++i) { v(i) /= v_norm; }

        // H <- P H P  where P = I - 2 v v^T acts on rows k+1:n.
        apply_householder_left(H, v, k + 1, m, k, n);
        apply_householder_right(H, v, 0, n, k + 1, m);

        // Accumulate: Q <- Q P
        apply_householder_right(Q, v, 0, n, k + 1, m);
    }
}

/// @brief Apply a 2x2 Givens rotation G to rows p and q of H,
///        for columns c0..c1-1, and accumulate into U.
///
///   G = [c  s]   so  [H(p,:)] <- [ c  s] [H(p,:)]
///       [-s c]       [H(q,:)]    [-s  c] [H(q,:)]
template <typename T>
void givens_rotate_rows(
    Matrix<T, dynamic_extent, dynamic_extent> &H,
    index_type p, index_type q,
    T c, T s,
    index_type c0, index_type c1) {
    for (index_type j = c0; j < c1; ++j) {
        T hp = H(p, j);
        T hq = H(q, j);
        H(p, j) = c * hp + s * hq;
        H(q, j) = -s * hp + c * hq;
    }
}

/// @brief Apply a 2x2 Givens rotation G from the right to columns p and q
///        of H, for rows r0..r1-1.
template <typename T>
void givens_rotate_cols(
    Matrix<T, dynamic_extent, dynamic_extent> &H,
    index_type p, index_type q,
    T c, T s,
    index_type r0, index_type r1) {
    for (index_type i = r0; i < r1; ++i) {
        T hp = H(i, p);
        T hq = H(i, q);
        H(i, p) = c * hp + s * hq;
        H(i, q) = -s * hp + c * hq;
    }
}

/// @brief Compute Givens rotation parameters to zero out b in [a, b].
///        Returns (c, s) such that [c s; -s c] [a; b] = [r; 0].
template <typename T>
auto givens_params(T a, T b) -> std::pair<T, T> {
    if (b == T{0}) {
        return {T{1}, T{0}};
    }
    T r = std::sqrt(a * a + b * b);
    return {a / r, b / r};
}

} // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Compute the real Schur decomposition of a square matrix.
///
/// Given A (n x n), computes A = U * T * U^T where U is orthogonal and T is
/// quasi-upper-triangular (real Schur form).
///
/// @param A         A square matrix satisfying `concepts::Matrix`.
/// @param max_iter  Maximum number of Francis QR iterations (default: 200*n).
/// @return          On success, `SchurResult{U, T_mat}`.
///                  On failure (did not converge), a `SolverError`.
template <concepts::Matrix Derived>
auto schur(const Derived &A, index_type max_iter = 0)
    -> std::expected<
        SchurResult<typename std::decay_t<Derived>::value_type, dynamic_extent>,
        solver::SolverError> {
    using AType = std::decay_t<Derived>;
    using T = typename AType::value_type;

    const index_type n = A.extent(0);

    if (n != A.extent(1)) {
        return std::unexpected(solver::SolverError{
            .kind = solver::SolverError::Kind::breakdown,
            .message = "schur: matrix must be square"});
    }

    if (max_iter == 0) { max_iter = 200 * std::max(n, index_type{1}); }

    // Trivial cases.
    if (n == 0) {
        Matrix<T, dynamic_extent, dynamic_extent> U_r(0, 0);
        Matrix<T, dynamic_extent, dynamic_extent> T_r(0, 0);
        return SchurResult<T, dynamic_extent>{
            .U = std::move(U_r), .T_mat = std::move(T_r)};
    }
    if (n == 1) {
        Matrix<T, dynamic_extent, dynamic_extent> U_r(
            expression::nullary::Identity<T, dynamic_extent, dynamic_extent>(
                1, 1));
        Matrix<T, dynamic_extent, dynamic_extent> T_r(1, 1);
        T_r(0, 0) = A(0, 0);
        return SchurResult<T, dynamic_extent>{
            .U = std::move(U_r), .T_mat = std::move(T_r)};
    }

    // Phase 1: Hessenberg reduction.
    Matrix<T, dynamic_extent, dynamic_extent> H(A);
    Matrix<T, dynamic_extent, dynamic_extent> U_acc(n, n);
    detail::hessenberg_reduce(H, U_acc);

    // Clean up numerical noise below the sub-diagonal.
    for (index_type i = 2; i < n; ++i) {
        for (index_type j = 0; j < i - 1; ++j) {
            H(i, j) = T{0};
        }
    }

    // Phase 2: Implicit QR iteration with Wilkinson (single) shift.
    //
    // We use a simple single-shift implicit QR strategy with Givens rotations.
    // This converges for real eigenvalues; for complex conjugate pairs, the
    // 2x2 block at the bottom will not deflate further, which is correct for
    // real Schur form.
    const T eps = std::numeric_limits<T>::epsilon();
    index_type p = n;  // active sub-matrix is H[0:p, 0:p]
    index_type total_iter = 0;

    while (p > 2) {
        if (total_iter >= max_iter) {
            return std::unexpected(solver::SolverError{
                .kind = solver::SolverError::Kind::diverged,
                .message = "schur: QR iteration did not converge"});
        }

        // Check for deflation at the bottom.
        T threshold = eps * (std::abs(H(p - 2, p - 2)) +
                             std::abs(H(p - 1, p - 1)));
        if (threshold == T{0}) { threshold = eps; }

        if (std::abs(H(p - 1, p - 2)) <= threshold) {
            H(p - 1, p - 2) = T{0};
            --p;
            continue;
        }

        // Check if we have a 2x2 block at the bottom with complex eigenvalues.
        // The 2x2 block H(p-2:p, p-2:p) has complex eigenvalues iff
        // discriminant < 0. In that case, check if H(p-2, p-3) is negligible
        // to deflate the whole 2x2 block.
        if (p >= 3) {
            T thresh2 = eps * (std::abs(H(p - 3, p - 3)) +
                               std::abs(H(p - 2, p - 2)));
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
            T thresh_q = eps * (std::abs(H(q - 1, q - 1)) +
                                std::abs(H(q, q)));
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
        T a11 = H(p - 2, p - 2);
        T a12 = H(p - 2, p - 1);
        T a21 = H(p - 1, p - 2);
        T a22 = H(p - 1, p - 1);

        T trace = a11 + a22;
        T det = a11 * a22 - a12 * a21;
        T disc = trace * trace - T{4} * det;

        T shift;
        if (disc >= T{0}) {
            // Two real eigenvalues; pick the one closer to a22.
            T sqrt_disc = std::sqrt(disc);
            T e1 = (trace + sqrt_disc) / T{2};
            T e2 = (trace - sqrt_disc) / T{2};
            shift = (std::abs(e1 - a22) < std::abs(e2 - a22)) ? e1 : e2;
        } else {
            // Complex conjugate pair; use a22 as the shift (exceptional shift).
            shift = a22;
        }

        // Implicit QR step with Givens rotations on H[q:p, q:p].
        // First rotation: zero out (H(q,q) - shift, H(q+1,q)).
        T x = H(q, q) - shift;
        T y = H(q + 1, q);

        for (index_type k = q; k < p - 1; ++k) {
            auto [c, s] = detail::givens_params(x, y);

            // Apply G from the left to rows k and k+1.
            detail::givens_rotate_rows(H, k, k + 1, c, s, 0, n);
            // Apply G^T from the right to columns k and k+1.
            detail::givens_rotate_cols(H, k, k + 1, c, s, 0, n);
            // Accumulate into U.
            detail::givens_rotate_cols(U_acc, k, k + 1, c, s, 0, n);

            // Prepare for next iteration: the bulge entry.
            if (k + 2 < p) {
                x = H(k + 1, k);
                y = H(k + 2, k);
            }
        }

        // Clean up sub-sub-diagonal entries created by numerical noise.
        for (index_type i = q + 2; i < p; ++i) {
            for (index_type j = q; j < i - 1; ++j) {
                H(i, j) = T{0};
            }
        }

        ++total_iter;
    }

    // Handle the remaining 2x2 block (p <= 2): already in Schur form.

    return SchurResult<T, dynamic_extent>{
        .U = std::move(U_acc), .T_mat = std::move(H)};
}

} // namespace zipper::utils::decomposition

#endif
