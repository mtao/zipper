/// @file tridiagonal_qr.hpp
/// @brief Eigenvalue decomposition of symmetric tridiagonal matrices via
///        implicit QL iteration with Wilkinson shift.
/// @ingroup eigenvalue
///
/// Given a symmetric tridiagonal matrix T (n x n), computes:
///
///   T = Q * diag(eigenvalues) * Q^T
///
/// where Q is orthogonal and eigenvalues are sorted in ascending order.
///
/// The algorithm is the standard implicit QL iteration with Wilkinson shift,
/// equivalent to LAPACK's DSTEQR. Each iteration applies a sequence of
/// Givens rotations to chase a bulge, converging sub-diagonal entries to zero.

#if !defined(ZIPPER_UTILS_EIGENVALUE_TRIDIAGONAL_QR_HPP)
#define ZIPPER_UTILS_EIGENVALUE_TRIDIAGONAL_QR_HPP

#include <algorithm>
#include <cmath>
#include <expected>
#include <limits>
#include <numeric>
#include <vector>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/utils/solver/result.hpp>

namespace zipper::utils::eigenvalue {

/// Result of a symmetric tridiagonal eigenvalue decomposition.
template <typename T, index_type N = dynamic_extent>
struct TridiagonalEigenResult {
    /// Eigenvalues in ascending order.
    Vector<T, N> eigenvalues;
    /// Orthogonal matrix whose columns are the corresponding eigenvectors.
    Matrix<T, N, N> eigenvectors;
};

/// @brief Compute eigenvalues and eigenvectors of a symmetric tridiagonal
///        matrix using the implicit QL algorithm with Wilkinson shift.
///
/// This is the standard algorithm for the symmetric tridiagonal eigenproblem,
/// equivalent to LAPACK's DSTEQR routine.
///
/// @param T_in      A symmetric tridiagonal matrix (only diagonal and
///                  sub-diagonal are read).
/// @param max_iter  Maximum number of QL sweeps per eigenvalue (default: 30).
/// @return          On success, eigenvalues (ascending) and eigenvectors.
///                  On failure, a SolverError.
template <concepts::Matrix Derived>
auto tridiagonal_qr_eigen(const Derived &T_in, index_type max_iter = 0)
    -> std::expected<
        TridiagonalEigenResult<typename std::decay_t<Derived>::value_type,
                               dynamic_extent>,
        solver::SolverError> {
    using AType = std::decay_t<Derived>;
    using T = typename AType::value_type;

    const index_type n = T_in.extent(0);

    if (n != T_in.extent(1)) {
        return std::unexpected(solver::SolverError{
            .kind = solver::SolverError::Kind::breakdown,
            .message = "tridiagonal_qr_eigen: matrix must be square"});
    }

    if (max_iter == 0) { max_iter = 30; }

    // Trivial cases.
    if (n == 0) {
        return TridiagonalEigenResult<T, dynamic_extent>{
            .eigenvalues = Vector<T, dynamic_extent>(0),
            .eigenvectors = Matrix<T, dynamic_extent, dynamic_extent>(0, 0)};
    }
    if (n == 1) {
        Vector<T, dynamic_extent> evals(1);
        evals(0) = T_in(0, 0);
        Matrix<T, dynamic_extent, dynamic_extent> evecs(
            expression::nullary::Identity<T, dynamic_extent, dynamic_extent>(
                1, 1));
        return TridiagonalEigenResult<T, dynamic_extent>{
            .eigenvalues = std::move(evals), .eigenvectors = std::move(evecs)};
    }

    // Extract diagonal (d) and sub-diagonal (e) into working arrays.
    VectorX<T> d(T_in.diagonal());
    VectorX<T> e(n); // e[0..n-2] are the sub-diagonal, e[n-1] = 0
    for (index_type i = 0; i + 1 < n; ++i) { e(i) = T_in(i + 1, i); }
    e(n - 1) = T{0};

    // Eigenvector accumulation.
    Matrix<T, dynamic_extent, dynamic_extent> Z(
        expression::nullary::Identity<T, dynamic_extent, dynamic_extent>(n, n));

    const T eps = std::numeric_limits<T>::epsilon();

    // Implicit QL iteration (from Numerical Recipes / Golub & Van Loan).
    // For each eigenvalue (indexed by l), we find the unreduced block
    // and apply shifted QL steps until the sub-diagonal converges.
    for (index_type l = 0; l < n; ++l) {
        index_type iter = 0;

        while (true) {
            // Find the smallest m >= l such that e[m] is negligible.
            index_type m = l;
            while (m + 1 < n) {
                T dd = std::abs(d(m)) + std::abs(d(m + 1));
                if (dd == T{0}) { dd = T{1}; }
                if (std::abs(e(m)) <= eps * dd) { break; }
                ++m;
            }

            // If m == l, the eigenvalue d[l] has converged.
            if (m == l) { break; }

            if (iter >= max_iter) {
                return std::unexpected(solver::SolverError{
                    .kind = solver::SolverError::Kind::diverged,
                    .message =
                        "tridiagonal_qr_eigen: QL iteration did not converge"});
            }
            ++iter;

            // Wilkinson shift: eigenvalue of the trailing 2x2 block
            // closest to d(l).
            T g = (d(l + 1) - d(l)) / (T{2} * e(l));
            T r = std::sqrt(g * g + T{1});
            // g = d(m) - d(l) + e(l) / (g + sign(g)*r)
            T sign_g = (g >= T{0}) ? T{1} : T{-1};
            g = d(m) - d(l) + e(l) / (g + sign_g * r);

            T s = T{1};
            T c = T{1};
            T p = T{0};

            // QL step: chase the bulge from m down to l+1.
            for (index_type i = m; i > l; --i) {
                T f = s * e(i - 1);
                T b = c * e(i - 1);

                // Compute Givens rotation.
                if (std::abs(f) >= std::abs(g)) {
                    c = g / f;
                    r = std::sqrt(c * c + T{1});
                    e(i) = f * r;
                    s = T{1} / r;
                    c *= s;
                } else {
                    s = f / g;
                    r = std::sqrt(s * s + T{1});
                    e(i) = g * r;
                    c = T{1} / r;
                    s *= c;
                }

                g = d(i) - p;
                r = (d(i - 1) - g) * s + T{2} * c * b;
                p = s * r;
                d(i) = g + p;
                g = c * r - b;

                // Accumulate eigenvectors: Z_new = Z * G(i-1, i, c, s).
                for (index_type k = 0; k < n; ++k) {
                    T zk_im1 = Z(k, i - 1);
                    T zk_i = Z(k, i);
                    Z(k, i - 1) = c * zk_im1 - s * zk_i;
                    Z(k, i) = s * zk_im1 + c * zk_i;
                }
            }

            d(l) -= p;
            e(l) = g;
            e(m) = T{0};
        }
    }

    // d already contains the eigenvalues; sort in ascending order and
    // permute eigenvectors accordingly.
    std::vector<index_type> perm(n);
    std::iota(perm.begin(), perm.end(), index_type{0});
    std::sort(perm.begin(), perm.end(), [&](index_type a, index_type b) {
        return d(a) < d(b);
    });

    Vector<T, dynamic_extent> sorted_evals(n);
    Matrix<T, dynamic_extent, dynamic_extent> sorted_evecs(n, n);
    for (index_type i = 0; i < n; ++i) {
        sorted_evals(i) = d(perm[i]);
        sorted_evecs.col(i) = Z.col(perm[i]);
    }

    return TridiagonalEigenResult<T, dynamic_extent>{
        .eigenvalues = std::move(sorted_evals),
        .eigenvectors = std::move(sorted_evecs)};
}

} // namespace zipper::utils::eigenvalue

#endif
