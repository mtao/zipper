/// @file givens.hpp
/// @brief Reusable Givens rotation utilities for decomposition routines.
/// @ingroup decompositions
///
/// Provides:
///   - `givens_params(a, b)` — compute (c, s) such that
///     [c s; -s c] [a; b] = [r; 0].
///   - `apply_givens_rows(M, p, q, c, s, c0, c1)` — apply rotation to rows
///     p and q of M for columns c0..c1-1.
///   - `apply_givens_cols(M, p, q, c, s, r0, r1)` — apply rotation to columns
///     p and q of M for rows r0..r1-1.
///
/// All functions are templated on concept-constrained types.

#if !defined(ZIPPER_UTILS_DECOMPOSITION_DETAIL_GIVENS_HPP)
#define ZIPPER_UTILS_DECOMPOSITION_DETAIL_GIVENS_HPP

#include <array>
#include <cmath>
#include <utility>

#include <zipper/Matrix.hpp>
#include <zipper/concepts/Matrix.hpp>
#include <zipper/types.hpp>

namespace zipper::utils::decomposition::detail {

/// @brief Compute Givens rotation parameters to zero out b in [a, b].
///
/// Returns (c, s) such that:
///   [c  s] [a]   [r]
///   [-s c] [b] = [0]
///
/// where r = sqrt(a^2 + b^2).
template <typename T>
auto givens_params(T a, T b) -> std::pair<T, T> {
    if (b == T{0}) { return {T{1}, T{0}}; }
    T r = std::sqrt(a * a + b * b);
    return {a / r, b / r};
}

/// @brief Apply a 2x2 Givens rotation from the left to rows p and q of M.
///
/// For columns c0..c1-1, computes:
///   [M(p,j)]     [ c  s] [M(p,j)]
///   [M(q,j)]  <- [-s  c] [M(q,j)]
///
/// @param M    Matrix to modify (must be mutable).
/// @param p    First row index.
/// @param q    Second row index.
/// @param c    Cosine of the rotation angle.
/// @param s    Sine of the rotation angle.
/// @param c0   First column of the affected range.
/// @param c1   One-past-last column of the affected range.
template <concepts::Matrix MDerived, typename T>
auto apply_givens_rows(MDerived &M,
                       index_type p,
                       index_type q,
                       T c,
                       T s,
                       index_type c0,
                       index_type c1) -> void {
    for (index_type j = c0; j < c1; ++j) {
        T hp = M(p, j);
        T hq = M(q, j);
        M(p, j) = c * hp + s * hq;
        M(q, j) = -s * hp + c * hq;
    }
}

/// @brief Apply a 2x2 Givens rotation from the right to columns p and q of M.
///
/// For rows r0..r1-1, computes:
///   [M(i,p), M(i,q)] <- [M(i,p), M(i,q)] [ c  -s]
///                                          [ s   c]
///
/// which is equivalent to:
///   M(i,p) =  c * M(i,p) + s * M(i,q)
///   M(i,q) = -s * M(i,p) + c * M(i,q)
///
/// @param M    Matrix to modify (must be mutable).
/// @param p    First column index.
/// @param q    Second column index.
/// @param c    Cosine of the rotation angle.
/// @param s    Sine of the rotation angle.
/// @param r0   First row of the affected range.
/// @param r1   One-past-last row of the affected range.
template <concepts::Matrix MDerived, typename T>
auto apply_givens_cols(MDerived &M,
                       index_type p,
                       index_type q,
                       T c,
                       T s,
                       index_type r0,
                       index_type r1) -> void {
    for (index_type i = r0; i < r1; ++i) {
        T hp = M(i, p);
        T hq = M(i, q);
        M(i, p) = c * hp + s * hq;
        M(i, q) = -s * hp + c * hq;
    }
}

} // namespace zipper::utils::decomposition::detail

#endif
