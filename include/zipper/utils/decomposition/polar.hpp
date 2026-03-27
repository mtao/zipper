/// @file polar.hpp
/// @brief SVD-based polar decomposition.
/// @ingroup decompositions
///
/// Given a square matrix F, computes the polar decomposition:
///
///   F = R * S
///
/// where R is a proper rotation (det R = +1, R^T R = I) and S is
/// symmetric positive semi-definite.
///
/// The decomposition is computed via SVD.  Given F = U * diag(sigma) * V^T:
///   R = U * V^T          (closest rotation to F)
///   S = V * diag(sigma) * V^T   (symmetric stretch)
///
/// If det(U * V^T) < 0 (improper rotation / reflection), the sign of the
/// column of U corresponding to the smallest singular value is flipped,
/// along with the corresponding singular value, to ensure det(R) = +1.
///
/// This approach is robust for singular and near-singular F (unlike the
/// iterative R_{k+1} = 0.5*(R_k + R_k^{-T}) method which requires
/// invertibility).

#if !defined(ZIPPER_UTILS_DECOMPOSITION_POLAR_HPP)
#define ZIPPER_UTILS_DECOMPOSITION_POLAR_HPP

#include <cmath>

#include <zipper/Matrix.hpp>
#include <zipper/concepts/Matrix.hpp>
#include <zipper/utils/decomposition/svd.hpp>
#include <zipper/utils/determinant.hpp>

namespace zipper::utils::decomposition {

// ─────────────────────────────────────────────────────────────────────────────
// Result type
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a polar decomposition F = R * S.
///
/// @tparam T  Scalar type.
/// @tparam N  Matrix dimension (static extent, or dynamic_extent).
template <typename T, index_type N>
struct PolarResult {
    using value_type = T;

    /// Proper rotation matrix (det R = +1, R^T R = I).
    Matrix<T, N, N> R;
    /// Symmetric positive semi-definite stretch matrix.
    Matrix<T, N, N> S;
};

// ─────────────────────────────────────────────────────────────────────────────
// Polar decomposition
// ─────────────────────────────────────────────────────────────────────────────

/// @brief SVD-based polar decomposition of a square matrix.
///
/// @param F  A square matrix.
/// @return   A `PolarResult` with rotation R and symmetric stretch S
///           such that F = R * S.
///
/// Polar decomposition always succeeds, so the result is returned directly
/// (not wrapped in `std::expected`).
template <concepts::Matrix Derived>
auto polar(const Derived &F) {
    using FType = std::decay_t<Derived>;
    using T = typename FType::value_type;
    constexpr index_type M = FType::extents_type::static_extent(0);
    constexpr index_type N = FType::extents_type::static_extent(1);

    // Polar decomposition requires a square matrix.
    static_assert(M == N || M == std::dynamic_extent
                      || N == std::dynamic_extent,
                  "polar decomposition requires a square matrix");

    const index_type n = F.extent(0);
    // Runtime check for dynamic extents.
    if (F.extent(0) != F.extent(1)) {
        // In practice this should be caught earlier, but guard anyway.
        // Return identity R and F as S as a fallback.
        PolarResult<T, M> result;
        result.R = Matrix<T, M, N>(n, n);
        result.S = Matrix<T, M, N>(n, n);
        for (index_type i = 0; i < n; ++i) {
            for (index_type j = 0; j < n; ++j) {
                result.R(i, j) = (i == j) ? T{1} : T{0};
                result.S(i, j) = F(i, j);
            }
        }
        return result;
    }

    auto [U, sigma, Vt] = svd(F);

    // R_candidate = U * Vt
    Matrix<T, M, N> R_candidate(U * Vt);

    // Check if R is a proper rotation (det > 0).
    T det_R = determinant(R_candidate);

    if (det_R < T{0}) {
        // Flip the last column of U (smallest singular value column,
        // since SVD returns singular values in descending order)
        // to get a proper rotation.
        const index_type last = n - 1;
        for (index_type i = 0; i < n; ++i) { U(i, last) = -U(i, last); }
        // Also flip the corresponding singular value.
        sigma(last) = -sigma(last);
        // Recompute R.
        R_candidate = Matrix<T, M, N>(U * Vt);
    }

    // S = R^T * F  (equivalently V * diag(sigma) * V^T, but R^T * F
    // is simpler and numerically equivalent).
    Matrix<T, M, N> S_result(R_candidate.transpose() * F);

    return PolarResult<T, M>{.R = std::move(R_candidate),
                             .S = std::move(S_result)};
}

} // namespace zipper::utils::decomposition

#endif
