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

#if !defined(ZIPPER_UTILS_DECOMPOSITION_POLAR_HPP)
#define ZIPPER_UTILS_DECOMPOSITION_POLAR_HPP

#include <cmath>
#include <concepts>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <zipper/Matrix.hpp>
#include <zipper/concepts/Matrix.hpp>
#include <zipper/utils/decomposition/svd.hpp>
#include <zipper/utils/decomposition/detail/shape_validation.hpp>
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

    /// Orthogonal factor (R^T R = I).
    Matrix<T, N, N> R;
    /// Symmetric stretch matrix.
    Matrix<T, N, N> S;
};

namespace detail {

template <bool Proper, concepts::Matrix Derived>
    requires StaticallySquare<Derived> &&
             std::floating_point<typename std::decay_t<Derived>::value_type>
auto polar_impl(const Derived &F)
    -> PolarResult<typename std::decay_t<Derived>::value_type,
                   std::decay_t<Derived>::extents_type::static_extent(0)> {
    using FType = std::decay_t<Derived>;
    using T = typename FType::value_type;
    constexpr index_type M = FType::extents_type::static_extent(0);
    constexpr index_type N = FType::extents_type::static_extent(1);

    const index_type n = F.extent(0);
    if (F.extent(0) != F.extent(1)) {
        throw std::invalid_argument("polar decomposition: matrix must be square");
    }

    auto [U, sigma, Vt] = svd(F);
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
/// Polar decomposition always succeeds, so the result is returned directly
/// (not wrapped in `std::expected`) for square inputs. A dynamically sized
/// nonsquare input throws `std::invalid_argument`.
template <concepts::Matrix Derived>
    requires detail::StaticallySquare<Derived> &&
             std::floating_point<typename std::decay_t<Derived>::value_type>
auto polar(const Derived &F)
    -> PolarResult<typename std::decay_t<Derived>::value_type,
                   std::decay_t<Derived>::extents_type::static_extent(0)> {
    return detail::polar_impl<false>(F);
}

/// @brief Orientation-preserving polar decomposition of a square matrix.
///
/// @return A `PolarResult` with proper orthogonal R (det R = +1) and signed
///         symmetric stretch S such that F = R * S. S may be indefinite.
/// @throws std::invalid_argument if a dynamically sized input is nonsquare.
template <concepts::Matrix Derived>
    requires detail::StaticallySquare<Derived> &&
             std::floating_point<typename std::decay_t<Derived>::value_type>
auto proper_polar(const Derived &F)
    -> PolarResult<typename std::decay_t<Derived>::value_type,
                   std::decay_t<Derived>::extents_type::static_extent(0)> {
    return detail::polar_impl<true>(F);
}

} // namespace zipper::utils::decomposition

#endif
