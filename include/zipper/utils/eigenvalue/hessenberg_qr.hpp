/// @file hessenberg_qr.hpp
/// @brief Eigenvalue extraction from upper Hessenberg matrices via
///        Francis implicit double-shift QR iteration.
/// @ingroup eigenvalue
///
/// Given an upper Hessenberg matrix H (n x n), computes the eigenvalues
/// (which may be complex for real matrices with complex eigenvalue pairs).
///
/// This builds on the Schur decomposition infrastructure in
/// `decomposition/schur.hpp` to obtain the quasi-upper-triangular (real Schur)
/// form, then extracts eigenvalues from the 1x1 and 2x2 diagonal blocks.
///
/// For eigenvector computation, we solve (T - lambda*I) * z = 0 via
/// back-substitution and map back through the Schur vectors U.

#if !defined(ZIPPER_UTILS_EIGENVALUE_HESSENBERG_QR_HPP)
#define ZIPPER_UTILS_EIGENVALUE_HESSENBERG_QR_HPP

#include <cmath>
#include <complex>
#include <expected>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/decomposition/schur.hpp>
#include <zipper/utils/solver/result.hpp>

namespace zipper::utils::eigenvalue {

/// Result of a general (possibly non-symmetric) eigenvalue decomposition.
template <typename T, index_type N = dynamic_extent>
struct EigenResult {
    /// Eigenvalues (may be complex).
    Vector<std::complex<T>, N> eigenvalues;
    /// Schur vectors (orthogonal matrix U from the real Schur decomposition).
    /// For full eigenvectors, one would need to solve (T - lambda*I)z = 0
    /// and compute U*z, but the Schur vectors are sufficient for many uses.
    Matrix<T, N, N> schur_vectors;
    /// Quasi-upper-triangular Schur form T.
    Matrix<T, N, N> schur_form;
};

/// @brief Extract eigenvalues from a quasi-upper-triangular (real Schur form)
///        matrix.
///
/// Scans the diagonal of T for 1x1 blocks (real eigenvalues) and 2x2 blocks
/// (complex conjugate pairs). A 2x2 block at position (i, i) is detected
/// when T(i+1, i) is non-negligible.
template <typename T>
auto extract_eigenvalues_from_schur(
    const Matrix<T, dynamic_extent, dynamic_extent> &T_mat)
    -> Vector<std::complex<T>, dynamic_extent> {
    const index_type n = T_mat.extent(0);
    Vector<std::complex<T>, dynamic_extent> evals(n);
    const T eps = std::numeric_limits<T>::epsilon();

    index_type i = 0;
    while (i < n) {
        if (i + 1 < n &&
            std::abs(T_mat(i + 1, i)) > eps * (std::abs(T_mat(i, i)) +
                                                 std::abs(T_mat(i + 1, i + 1)))) {
            // 2x2 block: eigenvalues of [[a, b], [c, d]]
            T a = T_mat(i, i);
            T b = T_mat(i, i + 1);
            T c = T_mat(i + 1, i);
            T d = T_mat(i + 1, i + 1);

            T trace = a + d;
            T det = a * d - b * c;
            T disc = trace * trace - T{4} * det;

            if (disc >= T{0}) {
                T sqrt_disc = std::sqrt(disc);
                evals(i) = std::complex<T>((trace + sqrt_disc) / T{2}, T{0});
                evals(i + 1) = std::complex<T>((trace - sqrt_disc) / T{2}, T{0});
            } else {
                T real_part = trace / T{2};
                T imag_part = std::sqrt(-disc) / T{2};
                evals(i) = std::complex<T>(real_part, imag_part);
                evals(i + 1) = std::complex<T>(real_part, -imag_part);
            }
            i += 2;
        } else {
            // 1x1 block: real eigenvalue
            evals(i) = std::complex<T>(T_mat(i, i), T{0});
            ++i;
        }
    }

    return evals;
}

/// @brief Compute eigenvalues (and Schur form) of a general square matrix.
///
/// Uses the real Schur decomposition (Hessenberg reduction + Francis QR)
/// to obtain the quasi-upper-triangular form, then extracts eigenvalues
/// from the diagonal blocks.
///
/// @param A         A square matrix.
/// @param max_iter  Maximum QR iterations (passed to schur()).
/// @return          On success, EigenResult with eigenvalues and Schur form.
///                  On failure, a SolverError.
template <concepts::Matrix Derived>
auto hessenberg_qr_eigen(const Derived &A, index_type max_iter = 0)
    -> std::expected<
        EigenResult<typename std::decay_t<Derived>::value_type, dynamic_extent>,
        solver::SolverError> {
    using AType = std::decay_t<Derived>;
    using T = typename AType::value_type;

    auto schur_result = decomposition::schur(A, max_iter);
    if (!schur_result) {
        return std::unexpected(schur_result.error());
    }

    auto eigenvalues = extract_eigenvalues_from_schur(schur_result->T_mat);

    return EigenResult<T, dynamic_extent>{
        .eigenvalues = std::move(eigenvalues),
        .schur_vectors = std::move(schur_result->U),
        .schur_form = std::move(schur_result->T_mat)};
}

} // namespace zipper::utils::eigenvalue

#endif
