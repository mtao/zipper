/// @file inverse.hpp
/// @brief Matrix inverse computation for 2x2, 3x3, and general n x n matrices.
/// @ingroup decompositions
///
/// Provides `zipper::utils::inverse(M)`, which dispatches to:
///
///   - **inverse2d(M)** — Closed-form 2x2 inverse using the cofactor formula:
///     A^{-1} = (1/det(A)) * adj(A).
///
///   - **inverse3d(M)** — Closed-form 3x3 inverse using the cofactor matrix
///     and determinant.
///
///   - **inverse_general(M)** — General n x n inverse via QR decomposition.
///     The algorithm factors A = Q * R, then solves R * x_j = Q^T * e_j for
///     each column j of the identity matrix using upper-triangular
///     back-substitution.  Complexity: O(n^3) for the QR factor plus O(n^3)
///     for n back-substitutions.
///
/// The runtime dispatch in `inverse()` checks `is_compatible<2,2>` and
/// `is_compatible<3,3>` on the matrix extents, falling back to the general
/// path for larger matrices.
///
/// @code
///   // 2x2 inverse (closed-form)
///   auto A = Matrix<double, 2, 2>({{1.0, 2.0}, {3.0, 4.0}});
///   auto A_inv = zipper::utils::inverse(A);
///
///   // 3x3 inverse (closed-form)
///   auto B = Matrix<double, 3, 3>({{1, 0, 0}, {0, 2, 0}, {0, 0, 3}});
///   auto B_inv = zipper::utils::inverse(B);
///
///   // General n x n inverse (QR-based)
///   auto C = Matrix<double, 4, 4>({...});
///   auto C_inv = zipper::utils::inverse(C);
/// @endcode
///
/// @throws std::runtime_error if the matrix is singular (determinant is zero
///         for 2x2/3x3, or R has a zero pivot for the general path).
///
/// @see zipper::utils::decomposition::qr — QR factorisation used by
///      inverse_general.
/// @see zipper::utils::solver::triangular_solve — back-substitution used by
///      inverse_general to solve each column.
/// @see zipper::expression::reductions::Determinant — determinant computation
///      used by inverse2d and inverse3d.
/// @see zipper::expression::unary::TriangularView — expression type wrapping
///      R for upper-triangular solve.

#if !defined(ZIPPER_UTILS_INVERSE_HPP)
#define ZIPPER_UTILS_INVERSE_HPP
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/concepts/Matrix.hpp>

#include "decomposition/qr.hpp"
#include "detail/tuple_to_array.hpp"
#include "zipper/utils/extents/is_compatible.hpp"
#include "zipper/detail/extents/swizzle_extents.hpp"
#include "zipper/expression/reductions/Determinant.hpp"
namespace zipper::utils {

namespace detail {

/// @brief Closed-form inverse of a 2x2 matrix.
///
/// Uses the cofactor formula:  A^{-1} = (1/det(A)) * adj(A), where
///   adj(A) = [[d, -b], [-c, a]]  for  A = [[a, b], [c, d]].
///
/// @param M  A matrix with extents compatible with 2x2.
/// @return   The inverse matrix with transposed static extents.
/// @throws   (via Determinant) No explicit throw, but division by zero will
///           produce Inf/NaN if the matrix is singular.
template <zipper::concepts::Matrix D>
auto inverse2d(const D& M) {
    zipper::utils::extents::throw_if_not_compatible<2, 2>(M.extents());
    auto extents = zipper::detail::extents::swizzle_extents<1, 0>(M.extents());
    using extents_type = std::decay_t<decltype(extents)>;

    using T = typename D::value_type;

    constexpr static index_type static_rows = extents_type::static_extent(0);
    constexpr static index_type static_cols = extents_type::static_extent(1);
    Matrix<T, static_rows, static_cols> I(extents);
    // a b
    // c d
    // 1/det ( d -b )
    //       ( -c a )
    const T& a = M(0, 0);
    const T& b = M(0, 1);
    const T& c = M(1, 0);
    const T& d = M(1, 1);
    T det = zipper::expression::reductions::Determinant(M.expression())();
    I(0, 0) = d / det;
    I(0, 1) = -b / det;
    I(1, 0) = -c / det;
    I(1, 1) = a / det;
    return I;
}

/// @brief Closed-form inverse of a 3x3 matrix using cofactors.
///
/// Computes each entry of the inverse as the cofactor of the transposed
/// position divided by the determinant:  (A^{-1})_{r,c} = C(c,r) / det(A).
///
/// @param M  A matrix with extents compatible with 3x3.
/// @return   The inverse matrix.
/// @throws   (via Determinant) No explicit throw, but division by zero will
///           produce Inf/NaN if the matrix is singular.
template <zipper::concepts::Matrix D>
auto inverse3d(const D& M) {
    zipper::utils::extents::throw_if_not_compatible<3, 3>(M.extents());
    auto extents = zipper::detail::extents::swizzle_extents<1, 0>(M.extents());
    using extents_type = std::decay_t<decltype(extents)>;

    using T = typename D::value_type;

    constexpr static index_type static_rows = extents_type::static_extent(0);
    constexpr static index_type static_cols = extents_type::static_extent(1);
    Matrix<T, static_rows, static_cols> I(extents);
    // a b
    // c d
    // 1/det ( d -b )
    //       ( -c a )
    T det = zipper::expression::reductions::Determinant(M.expression())();
    auto cofactor = [](const auto& v_, index_type r, index_type c) {
        auto det2 = [](const auto& v, index_type r0, index_type c0,
                       index_type r1, index_type c1) {
            const auto val = zipper::expression::reductions::detail::det2(
                v(r0, c0), v(r0, c1), v(r1, c0), v(r1, c1));
            return val;
        };
        return det2(v_, (r + 1) % 3, (c + 1) % 3, (r + 2) % 3, (c + 2) % 3);
    };
    for (index_type r = 0; r < 3; ++r) {
        for (index_type c = 0; c < 3; ++c) {
            I(r, c) = cofactor(M, c, r) / det;
        }
    }
    return I;
}

/// @brief General n x n matrix inverse via QR decomposition.
///
/// Computes A^{-1} by factoring A = Q * R, then solving
/// R * x_j = Q^T * e_j  for each column j of the identity.
///
/// @throws std::runtime_error if A is singular (R has a zero pivot).
template <zipper::concepts::Matrix D>
auto inverse_general(const D& M) {
    using T = typename D::value_type;
    constexpr index_type N = D::extents_type::static_extent(0);

    const index_type n = M.extent(0);

    // Factor A = Q * R.
    auto [Q, R] = decomposition::qr(M);

    // Build upper-triangular view of R (which is n x n for a square matrix).
    auto R_upper = expression::triangular_view<
        expression::TriangularMode::Upper>(R);

    // Result matrix: same static extents as M^T.
    auto extents = zipper::detail::extents::swizzle_extents<1, 0>(M.extents());
    using extents_type = std::decay_t<decltype(extents)>;
    constexpr index_type static_rows = extents_type::static_extent(0);
    constexpr index_type static_cols = extents_type::static_extent(1);
    Matrix<T, static_rows, static_cols> Inv(extents);

    // Solve R * x_j = Q^T * e_j  for each column j.
    for (index_type j = 0; j < n; ++j) {
        // Compute c = Q^T * e_j = column j of Q^T = row j of Q... no.
        // c(i) = sum_k Q(k, i) * e_j(k) = Q(j, i).
        Vector<T, N> c(n);
        for (index_type i = 0; i < n; ++i) {
            c(i) = Q(j, i);
        }

        auto solve_result = solver::triangular_solve(R_upper, c);
        if (!solve_result) {
            throw std::runtime_error(
                "inverse: matrix is singular — " +
                solve_result.error().message);
        }

        auto& x_j = *solve_result;
        for (index_type i = 0; i < n; ++i) {
            Inv(i, j) = x_j(i);
        }
    }

    return Inv;
}

}  // namespace detail

/// @brief Compute the inverse of a square matrix.
///
/// Dispatches to the optimal implementation based on matrix size:
///   - 2x2: closed-form cofactor formula (O(1) flops).
///   - 3x3: cofactor matrix divided by determinant (O(1) flops).
///   - n x n (n > 3): QR decomposition + n back-substitutions (O(n^3) flops).
///
/// @param d  A square matrix satisfying `concepts::Matrix`.
/// @return   The inverse matrix (same value_type, transposed static extents).
/// @throws   std::runtime_error if the matrix is singular.
///
/// @code
///   auto A = Matrix<double, 3, 3>({{2, 0, 0}, {0, 3, 0}, {0, 0, 4}});
///   auto A_inv = zipper::utils::inverse(A);
///   // A_inv(0,0) == 0.5, A_inv(1,1) == 1/3, A_inv(2,2) == 0.25
/// @endcode
template <zipper::concepts::Matrix D>
auto inverse(const D& d) {
    const auto& extents = d.extents();
    if (zipper::utils::extents::is_compatible<2, 2>(extents)) {
        return detail::inverse2d(d);
    } else if (zipper::utils::extents::is_compatible<3, 3>(extents)) {
        return detail::inverse3d(d);
    }
    return detail::inverse_general(d);
}
}  // namespace zipper::utils
#endif
