/// @file householder.hpp
/// @brief Reusable Householder reflection utilities for decomposition routines.
/// @ingroup decompositions
///
/// Provides:
///   - `householder_vector(x)` — compute the Householder vector and
///     reflection constant for a given input vector.
///   - `apply_householder_left(M, v, r0, c0, c1)` — apply P = I - 2vv^T from
///     the left to a sub-block of M.
///   - `apply_householder_right(M, v, r0, r1, c0)` — apply P from the right
///     to a sub-block of M.
///
/// All functions are templated on concept-constrained types so they work on
/// any MatrixBase/VectorBase expression, not just concrete owning types.

#if !defined(ZIPPER_UTILS_DECOMPOSITION_DETAIL_HOUSEHOLDER_HPP)
#define ZIPPER_UTILS_DECOMPOSITION_DETAIL_HOUSEHOLDER_HPP

#include <cmath>
#include <limits>
#include <optional>

#include <zipper/Vector.hpp>
#include <zipper/concepts/Matrix.hpp>
#include <zipper/concepts/Vector.hpp>
#include <zipper/types.hpp>

namespace zipper::utils::decomposition::detail {

/// Result of computing a Householder vector from an input vector x.
///
/// The Householder reflection P = I - 2 v v^T satisfies P x = sigma * e_0.
/// `v` is unit-length. If the input was too small (near-zero), the result
/// is std::nullopt.
template <typename T, index_type N>
struct HouseholderVector {
    /// Unit Householder vector.
    Vector<T, N> v;
    /// The resulting scalar: P * x = sigma * e_0.
    T sigma;
};

/// @brief Compute the Householder vector for a given input vector.
///
/// Given x, computes unit vector v such that (I - 2vv^T) x = sigma * e_0,
/// choosing the sign of sigma to avoid cancellation.
///
/// @param x  Input vector of length m.
/// @return   HouseholderVector on success, or std::nullopt if ||x|| is
///           negligible.
template <concepts::Vector VDerived>
auto householder_vector(const VDerived &x) -> std::optional<
    HouseholderVector<typename std::decay_t<VDerived>::value_type,
                      std::decay_t<VDerived>::extents_type::static_extent(0)>> {
    using T = typename std::decay_t<VDerived>::value_type;
    constexpr index_type N =
        std::decay_t<VDerived>::extents_type::static_extent(0);

    // Copy x into an owning vector for modification.
    Vector<T, N> v(x);

    T sigma = v.norm();
    if (sigma < std::numeric_limits<T>::epsilon()) { return std::nullopt; }

    // Choose sign to avoid cancellation: v(0) -= sign(v(0)) * ||x||.
    if (v(0) >= T{0}) { sigma = -sigma; }

    v(0) -= sigma;

    T v_norm = v.norm();
    if (v_norm < std::numeric_limits<T>::min()) { return std::nullopt; }
    v.normalize();

    return HouseholderVector<T, N>{.v = std::move(v), .sigma = sigma};
}

/// @brief Apply a Householder reflection from the left to a sub-block of M.
///
/// Performs: M[r0:r0+len, c0:c1) -= 2 * v * (v^T * M[r0:r0+len, c0:c1))
///
/// where v has length `len` and indexes rows r0..r0+len-1 of M.
/// This updates M in place, column by column.
///
/// @param M    Matrix to modify (must be mutable).
/// @param v    Unit Householder vector of length len.
/// @param r0   First row of the affected block.
/// @param c0   First column of the affected block.
/// @param c1   One-past-last column of the affected block.
template <concepts::Matrix MDerived, concepts::Vector VDerived>
auto apply_householder_left(MDerived &M,
                            const VDerived &v,
                            index_type r0,
                            index_type c0,
                            index_type c1) -> void {
    using T = typename std::decay_t<VDerived>::value_type;
    const index_type len = v.extent(0);

    // For each column j, compute dot = v^T * M(r0:r0+len, j),
    // then M(r0+i, j) -= 2 * v(i) * dot.
    for (index_type j = c0; j < c1; ++j) {
        auto col = M.col(j);
        auto col_sub = col.segment(r0, len);
        auto dot = v.dot(col_sub);
        col_sub -= (T{2} * dot) * v;
    }
}

/// @brief Apply a Householder reflection from the right to a sub-block of M.
///
/// Performs: M[r0:r1, c0:c0+len) -= 2 * (M[r0:r1, c0:c0+len) * v) * v^T
///
/// where v has length `len` and indexes columns c0..c0+len-1 of M.
/// This updates M in place, row by row.
///
/// @param M    Matrix to modify (must be mutable).
/// @param v    Unit Householder vector of length len.
/// @param r0   First row of the affected block.
/// @param r1   One-past-last row of the affected block.
/// @param c0   First column of the affected block.
template <concepts::Matrix MDerived, concepts::Vector VDerived>
auto apply_householder_right(MDerived &M,
                             const VDerived &v,
                             index_type r0,
                             index_type r1,
                             index_type c0) -> void {
    using T = typename std::decay_t<VDerived>::value_type;
    const index_type len = v.extent(0);

    // For each row i, compute dot = M(i, c0:c0+len) . v,
    // then M(i, c0+j) -= 2 * dot * v(j).
    for (index_type i = r0; i < r1; ++i) {
        auto row = M.row(i);
        auto row_sub = row.segment(c0, len);
        auto dot = v.dot(row_sub);
        row_sub -= (T{2} * dot) * v;
    }
}

} // namespace zipper::utils::decomposition::detail

#endif
