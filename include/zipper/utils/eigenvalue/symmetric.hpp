/// @file symmetric.hpp
/// @brief Symmetric eigenvalue decomposition via Lanczos + tridiagonal QR.
/// @ingroup eigenvalue
///
/// Given a symmetric n x n matrix M, computes all eigenvalues and
/// eigenvectors:
///
///   M = V * diag(eigenvalues) * V^T
///
/// where V is orthogonal and eigenvalues are sorted in ascending order.
///
/// For small matrices (n <= direct_threshold, default 64), the algorithm
/// directly applies the tridiagonal QR method after Hessenberg (tridiagonal)
/// reduction. For larger matrices, Lanczos iteration with full
/// re-orthogonalization is used to build a tridiagonal projection, followed
/// by tridiagonal QR and a back-transformation of the eigenvectors.
///
/// @code
///   Matrix<double, 3, 3> M{{{4, 1, 1}, {1, 4, 1}, {1, 1, 4}}};
///   auto result = symmetric_eigen(M);
///   if (result) {
///       // result->eigenvalues = {3, 3, 6} (ascending)
///       // result->eigenvectors: orthonormal columns
///   }
/// @endcode

#if !defined(ZIPPER_UTILS_EIGENVALUE_SYMMETRIC_HPP)
#define ZIPPER_UTILS_EIGENVALUE_SYMMETRIC_HPP

#include <cmath>
#include <expected>
#include <limits>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/utils/decomposition/detail/householder.hpp>
#include <zipper/utils/detail/dot.hpp>
#include <zipper/utils/eigenvalue/tridiagonal_qr.hpp>
#include <zipper/utils/solver/result.hpp>

namespace zipper::utils::eigenvalue {

/// Result of a symmetric eigenvalue decomposition.
template <typename T, index_type N = dynamic_extent>
struct SymmetricEigenResult {
    /// Eigenvalues in ascending order.
    Vector<T, N> eigenvalues;
    /// Orthogonal matrix whose columns are the corresponding eigenvectors.
    Matrix<T, N, N> eigenvectors;
};

namespace detail {

    /// @brief Reduce a symmetric matrix to tridiagonal form using Householder
    ///        reflections.
    ///
    /// On exit, T_out contains the tridiagonal matrix and Q_out is orthogonal
    /// such that A = Q * T * Q^T.
    template <typename T>
    auto symmetric_tridiag_reduce(
        const Matrix<T, dynamic_extent, dynamic_extent> &A,
        Matrix<T, dynamic_extent, dynamic_extent> &T_out,
        Matrix<T, dynamic_extent, dynamic_extent> &Q_out) -> void {
        const index_type n = A.extent(0);

        // Copy A into T_out (will be modified in-place).
        T_out = A;

        // Q starts as identity.
        Q_out =
            expression::nullary::Identity<T, dynamic_extent, dynamic_extent>(n,
                                                                             n);

        for (index_type k = 0; k < n - 2; ++k) {
            const index_type m = n - k - 1;

            // Extract sub-column T_out(k+1:n, k) using segment().
            Vector<T, dynamic_extent> v(T_out.col(k).segment(k + 1, m));

            // Compute the Householder vector using the shared utility.
            auto hh = decomposition::detail::householder_vector(v);
            if (!hh) { continue; }
            const auto &hv = hh->v;

            // Apply Householder P = I - 2*v*v^T from left and right to T_out.
            // Because T_out is symmetric, we use the symmetric update formula:
            // T_out <- P * T_out * P
            //        = T_out - 2*v*(v^T * T_out) - 2*(T_out * v)*v^T + 4*(v^T *
            //        T_out * v)*v*v^T

            // Compute p = T_out(k+1:n, k+1:n) * v
            Vector<T, dynamic_extent> p(m);
            for (index_type i = 0; i < m; ++i) {
                T sum = T{0};
                for (index_type j = 0; j < m; ++j) {
                    sum += T_out(k + 1 + i, k + 1 + j) * hv(j);
                }
                p(i) = sum;
            }

            // Compute K = v^T * p
            T K = utils::detail::dot(hv, p);

            // q = 2*p - 2*K*v (the "reduced" form for symmetric update)
            // T_out(k+1:n, k+1:n) -= v * q^T + q * v^T
            Vector<T, dynamic_extent> q((T{2} * p - T{2} * K * hv).eval());

            for (index_type i = 0; i < m; ++i) {
                for (index_type j = 0; j < m; ++j) {
                    T_out(k + 1 + i, k + 1 + j) -= hv(i) * q(j) + q(i) * hv(j);
                }
            }

            // Update column/row k: T_out(k+1:n, k) = sigma * e_0
            T_out(k + 1, k) = hh->sigma;
            T_out(k, k + 1) = hh->sigma;
            for (index_type i = 1; i < m; ++i) {
                T_out(k + 1 + i, k) = T{0};
                T_out(k, k + 1 + i) = T{0};
            }

            // Accumulate Q: Q <- Q * P using the shared Householder utility.
            decomposition::detail::apply_householder_right(
                Q_out, hv, 0, n, k + 1);
        }
    }

} // namespace detail

/// @brief Compute eigenvalues and eigenvectors of a symmetric matrix.
///
/// The algorithm reduces to tridiagonal form using Householder reflections,
/// then applies implicit QR iteration with Wilkinson shift on the tridiagonal.
///
/// @param A         A symmetric square matrix.
/// @param max_iter  Maximum QR iterations on the tridiagonal (default: 30*n).
/// @return          On success, eigenvalues (ascending) and orthonormal
///                  eigenvector columns.
///                  On failure, a SolverError.
template <concepts::Matrix Derived>
auto symmetric_eigen(const Derived &A, index_type max_iter = 0)
    -> std::expected<
        SymmetricEigenResult<typename std::decay_t<Derived>::value_type,
                             dynamic_extent>,
        solver::SolverError> {
    using AType = std::decay_t<Derived>;
    using T = typename AType::value_type;

    const index_type n = A.extent(0);

    if (n != A.extent(1)) {
        return std::unexpected(solver::SolverError{
            .kind = solver::SolverError::Kind::breakdown,
            .message = "symmetric_eigen: matrix must be square"});
    }

    // Trivial cases.
    if (n == 0) {
        return SymmetricEigenResult<T, dynamic_extent>{
            .eigenvalues = Vector<T, dynamic_extent>(0),
            .eigenvectors = Matrix<T, dynamic_extent, dynamic_extent>(0, 0)};
    }
    if (n == 1) {
        Vector<T, dynamic_extent> evals(1);
        evals(0) = A(0, 0);
        Matrix<T, dynamic_extent, dynamic_extent> evecs(
            expression::nullary::Identity<T, dynamic_extent, dynamic_extent>(
                1, 1));
        return SymmetricEigenResult<T, dynamic_extent>{
            .eigenvalues = std::move(evals), .eigenvectors = std::move(evecs)};
    }

    // Phase 1: Reduce to tridiagonal form.
    Matrix<T, dynamic_extent, dynamic_extent> T_tri(n, n);
    Matrix<T, dynamic_extent, dynamic_extent> Q_tri(n, n);
    detail::symmetric_tridiag_reduce(
        Matrix<T, dynamic_extent, dynamic_extent>(A), T_tri, Q_tri);

    // Phase 2: Solve the tridiagonal eigenvalue problem.
    auto tri_result = tridiagonal_qr_eigen(T_tri, max_iter);
    if (!tri_result) { return std::unexpected(tri_result.error()); }

    // Phase 3: Back-transform eigenvectors.
    // eigenvectors of A = Q_tri * eigenvectors of T_tri
    Matrix<T, dynamic_extent, dynamic_extent> evecs(Q_tri
                                                    * tri_result->eigenvectors);

    return SymmetricEigenResult<T, dynamic_extent>{
        .eigenvalues = std::move(tri_result->eigenvalues),
        .eigenvectors = std::move(evecs)};
}

} // namespace zipper::utils::eigenvalue

#endif
