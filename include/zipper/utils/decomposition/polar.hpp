/// @file polar.hpp
/// @brief SVD-based polar decomposition.
/// @ingroup decompositions
///
/// Given a square matrix F, computes the polar decomposition:
///
///   F = R * S
///
/// where R is orthogonal (R^T R = I) and S is symmetric positive
/// semi-definite.
///
/// The decomposition is computed via SVD.  Given F = U * diag(sigma) * V^T:
///   R = U * V^T          (closest orthogonal matrix to F)
///   S = V * diag(sigma) * V^T   (symmetric stretch)
///
/// `proper_polar` instead forces det(R) = +1 by moving a reflection into a
/// signed symmetric stretch.
///
/// This intentionally changes `polar` from its former orientation-preserving
/// behavior to the standard polar-decomposition semantics. Call
/// `proper_polar` when an orientation-preserving orthogonal factor is required.
///
/// This approach is robust for singular and near-singular F (unlike the
/// iterative R_{k+1} = 0.5*(R_k + R_k^{-T}) method which requires
/// invertibility).
/// The scalar type must be a field closed under square root; ordinary exact
/// rational types require a promoted or approximate square-root type.

#if !defined(ZIPPER_UTILS_DECOMPOSITION_POLAR_HPP)
#define ZIPPER_UTILS_DECOMPOSITION_POLAR_HPP

#include <expected>
#include <type_traits>
#include <utility>

#include <zipper/Matrix.hpp>
#include <zipper/concepts/Matrix.hpp>
#include <zipper/utils/decomposition/detail/scalar_math.hpp>
#include <zipper/utils/decomposition/detail/shape_validation.hpp>
#include <zipper/utils/decomposition/svd.hpp>
#include <zipper/utils/determinant.hpp>
#include <zipper/utils/solver/error.hpp>

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

    /// Orthogonal factor (R^T R = I).
    Matrix<T, N, N> R;
    /// Symmetric stretch matrix.
    Matrix<T, N, N> S;
};

namespace detail {

template <bool Proper, concepts::Matrix Derived>
    requires StaticallySquare<Derived> &&
              orthogonal_decomposition_scalar<
                  typename std::decay_t<Derived>::value_type>
auto polar_impl(const Derived &F)
    -> std::expected<
        PolarResult<typename std::decay_t<Derived>::value_type,
                    std::decay_t<Derived>::extents_type::static_extent(0)>,
        solver::SolverError> {
    using FType = std::decay_t<Derived>;
    using T = typename FType::value_type;
    constexpr index_type M = FType::extents_type::static_extent(0);
    constexpr index_type N = FType::extents_type::static_extent(1);

    const index_type n = F.extent(0);
    if (auto valid = validate_square(F, "polar decomposition"); !valid) {
        return std::unexpected(std::move(valid.error()));
    }

    auto svd_result = svd(F);
    if (!svd_result) {
        return std::unexpected(std::move(svd_result.error()));
    }
    auto &[U, sigma, Vt] = *svd_result;
    Matrix<T, M, N> R(U * Vt);

    if constexpr (Proper) {
        if (determinant(R) < T{0}) {
            const index_type last = n - 1;
            U.col(last) = -U.col(last);
            R = Matrix<T, M, N>(U * Vt);
        }
    }

    Matrix<T, M, N> S(R.transpose() * F);
    return PolarResult<T, M>{.R = std::move(R), .S = std::move(S)};
}

} // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
// Polar decomposition
// ─────────────────────────────────────────────────────────────────────────────

/// @brief SVD-based polar decomposition of a square matrix.
///
/// @param F  A square matrix.
/// @return   A `PolarResult` with orthogonal R and symmetric positive
///           semi-definite stretch S such that F = R * S.
///
/// @return   `std::expected<PolarResult, SolverError>`. Errors:
///           `invalid_input` for a dynamically sized nonsquare or NaN input,
///           `diverged` if the underlying SVD does not converge.
template <concepts::Matrix Derived>
    requires detail::StaticallySquare<Derived> &&
              detail::orthogonal_decomposition_scalar<
                  typename std::decay_t<Derived>::value_type>
auto polar(const Derived &F)
    -> std::expected<
        PolarResult<typename std::decay_t<Derived>::value_type,
                    std::decay_t<Derived>::extents_type::static_extent(0)>,
        solver::SolverError> {
    return detail::polar_impl<false>(F);
}

/// @brief Orientation-preserving polar decomposition of a square matrix.
///
/// @return A `PolarResult` with proper orthogonal R (det R = +1) and signed
///         symmetric stretch S such that F = R * S. S may be indefinite.
///         or a `SolverError` (see `polar`).
template <concepts::Matrix Derived>
    requires detail::StaticallySquare<Derived> &&
              detail::orthogonal_decomposition_scalar<
                  typename std::decay_t<Derived>::value_type>
auto proper_polar(const Derived &F)
    -> std::expected<
        PolarResult<typename std::decay_t<Derived>::value_type,
                    std::decay_t<Derived>::extents_type::static_extent(0)>,
        solver::SolverError> {
    return detail::polar_impl<true>(F);
}

} // namespace zipper::utils::decomposition

#endif
