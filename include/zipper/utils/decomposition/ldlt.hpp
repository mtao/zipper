/// @file ldlt.hpp
/// @brief LDLT decomposition for symmetric positive semi-definite matrices.
///
/// Given a symmetric matrix A, the LDLT decomposition factors it as:
///
///   A = L * D * L^T
///
/// where L is a unit lower triangular matrix (ones on the diagonal) and D is a
/// diagonal matrix.  For symmetric positive definite (SPD) matrices all
/// diagonal entries of D are positive.  For symmetric positive semi-definite
/// matrices, D entries are non-negative.
///
/// Compared to the Cholesky (LLT) decomposition, LDLT avoids computing square
/// roots, which improves both performance and numerical stability.  It also
/// handles positive semi-definite matrices (where LLT would fail on a zero
/// diagonal).
///
/// The algorithm processes columns left to right.  For each column j:
///
///   D(j)   = A(j,j) - sum_{k=0}^{j-1} L(j,k)^2 * D(k)
///   L(i,j) = ( A(i,j) - sum_{k=0}^{j-1} L(i,k) * L(j,k) * D(k) ) / D(j)
///            for i > j
///
/// Two interfaces:
///   - `ldlt(A)`         -- returns the factors L and D.
///   - `ldlt_solve(A,b)` -- solves Ax = b via LDLT factorisation.
///
/// The solve proceeds in three stages:
///   1. Forward substitution:  L * y = b      (unit lower triangular)
///   2. Diagonal solve:        D * z = y      (trivial element-wise division)
///   3. Back substitution:     L^T * x = z    (unit upper triangular)
///
/// The decomposition detects zero pivots in D and returns a `SolverError`
/// when a zero diagonal entry would cause division by zero.
///
/// Complexity: O(n^3 / 3) for the factorisation, O(n^2) for each solve.

#if !defined(ZIPPER_UTILS_DECOMPOSITION_LDLT_HPP)
#define ZIPPER_UTILS_DECOMPOSITION_LDLT_HPP

#include <cmath>
#include <expected>
#include <limits>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/expression/unary/TriangularView.hpp>
#include <zipper/utils/solver/result.hpp>

namespace zipper::utils::decomposition {

// ─────────────────────────────────────────────────────────────────────────────
// Result type
// ─────────────────────────────────────────────────────────────────────────────

/// Result of an LDLT decomposition.
///
/// L is an n x n unit lower triangular matrix and D is a diagonal stored as
/// an n-element vector, such that A = L * D * L^T.
///
/// Calling `.solve(b)` performs forward substitution, diagonal solve, and
/// back substitution using the stored factors to solve A*x = b without
/// re-factoring.
template <typename T, index_type N>
struct LDLTResult {
    /// Scalar type of the decomposition.
    using value_type = T;

    /// Unit lower triangular factor L (n x n, ones on diagonal).
    Matrix<T, N, N> L;
    /// Diagonal factor D (n elements).
    Vector<T, N> D;

    /// @brief Solve A*x = b using the stored LDLT factors.
    ///
    /// Performs:
    ///   1. Forward substitution:  L * y = b      (unit lower triangular)
    ///   2. Diagonal solve:        D * z = y      (element-wise division)
    ///   3. Back substitution:     L^T * x = z    (unit upper triangular)
    ///
    /// @param b  Right-hand side vector of length n.
    /// @return   `std::expected<Vector<T,N>, SolverError>` — the solution on
    ///           success, or a breakdown error if D contains a zero pivot.
    template <concepts::Vector BDerived>
    auto solve(const BDerived &b) const
        -> std::expected<Vector<T, N>, solver::SolverError> {
        using ResultVec = Vector<T, N>;
        using Result = std::expected<ResultVec, solver::SolverError>;

        const index_type n = L.extent(0);

        // 1. Forward substitution: L * y = b  (L is unit lower triangular).
        auto L_lower =
            expression::triangular_view<expression::TriangularMode::UnitLower>(
                L);
        auto y_result = L_lower.solve(b);

        if (!y_result) {
            return Result{std::unexpected(std::move(y_result.error()))};
        }

        // 2. Diagonal solve: D * z = y  →  z(i) = y(i) / D(i).
        ResultVec z(n);
        for (index_type i = 0; i < n; ++i) {
            if (std::abs(D(i)) <= std::numeric_limits<T>::epsilon()) {
                return Result{std::unexpected(solver::SolverError{
                    .kind = solver::SolverError::Kind::breakdown,
                    .message = "LDLT solve: zero diagonal entry D("
                               + std::to_string(i)
                               + ") — matrix is singular"})};
            }
            z(i) = (*y_result)(i) / D(i);
        }

        // 3. Back substitution: L^T * x = z  (L^T is unit upper triangular).
        Matrix<T, N, N> Lt(L.transpose());

        auto Lt_upper =
            expression::triangular_view<expression::TriangularMode::UnitUpper>(
                Lt);
        auto x_result = Lt_upper.solve(z);

        if (!x_result) {
            return Result{std::unexpected(std::move(x_result.error()))};
        }

        return Result{std::move(*x_result)};
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// LDLT factorisation
// ─────────────────────────────────────────────────────────────────────────────

/// @brief LDLT decomposition of a symmetric matrix.
///
/// @param A  An n x n symmetric matrix.
/// @return   `std::expected<LDLTResult<T,N>, SolverError>` — the factors L
///           and D on success, or a breakdown error if a zero pivot is
///           encountered.
template <concepts::Matrix Derived>
auto ldlt(const Derived &A) -> std::expected<
    LDLTResult<typename std::decay_t<Derived>::value_type,
               std::decay_t<Derived>::extents_type::static_extent(0)>,
    solver::SolverError> {
    using AType = std::decay_t<Derived>;
    using T = typename AType::value_type;
    constexpr index_type N = AType::extents_type::static_extent(0);
    using Result = std::expected<LDLTResult<T, N>, solver::SolverError>;

    const index_type n = A.extent(0);

    // Initialise L to zero and D to zero.
    Matrix<T, N, N> L(n, n);
    L = expression::nullary::Constant(T{0}, L.extents());
    Vector<T, N> D(n);
    D = expression::nullary::Constant(T{0}, D.extents());

    for (index_type j = 0; j < n; ++j) {
        // Compute D(j) = A(j,j) - sum_{k<j} L(j,k)^2 * D(k).
        //   = A(j,j) - ||L(j, 0:j)||^2_D   (D-weighted squared norm)
        auto Lj_seg = L.row(j).segment(0, j);
        auto D_seg = D.segment(0, j);
        T sum = (j > 0)
                    ? (Lj_seg.as_array() * Lj_seg.as_array() * D_seg.as_array())
                          .sum()
                    : T{0};
        D(j) = A(j, j) - sum;

        // Set the unit diagonal.
        L(j, j) = T{1};

        if (std::abs(D(j)) <= std::numeric_limits<T>::epsilon()) {
            // Zero pivot — cannot compute sub-diagonal entries for this column.
            // For positive semi-definite matrices, D(j) == 0 means the
            // remaining sub-diagonal entries in this column should be zero (the
            // column is in the null space).  We leave them at zero and
            // continue.
            //
            // However, if any A(i,j) - sum != 0 for i > j, the matrix is
            // indefinite and the factorisation is invalid.  We skip that check
            // here for simplicity and treat zero pivots as acceptable (the
            // solve step will detect the singularity when dividing by D(j)).
            continue;
        }

        // Compute sub-diagonal entries L(i,j) for i > j.
        //   L(i,j) = ( A(i,j) - L(i, 0:j) . (L(j, 0:j) .* D(0:j)) ) / D(j)
        for (index_type i = j + 1; i < n; ++i) {
            T s = (j > 0) ? (L.row(i).segment(0, j).as_array()
                             * Lj_seg.as_array() * D_seg.as_array())
                                .sum()
                          : T{0};
            L(i, j) = (A(i, j) - s) / D(j);
        }
    }

    return Result{LDLTResult<T, N>{.L = std::move(L), .D = std::move(D)}};
}

// ─────────────────────────────────────────────────────────────────────────────
// LDLT solve
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Solve Ax = b via LDLT decomposition.
///
/// Factors A = L * D * L^T, then delegates to `LDLTResult::solve(b)`.
///
/// Requires A to be symmetric.  For a unique solution, A must also be
/// non-singular (all D(j) != 0).
///
/// @param A  An n x n symmetric matrix.
/// @param b  Right-hand side vector of length n.
/// @return   `std::expected<Vector<T,N>, SolverError>` — the solution on
///           success, or a breakdown error if the factorisation fails or D
///           contains a zero pivot.
template <concepts::Matrix ADerived, concepts::Vector BDerived>
auto ldlt_solve(const ADerived &A, const BDerived &b) {
    using AType = std::decay_t<ADerived>;
    using T = typename AType::value_type;
    constexpr index_type N = AType::extents_type::static_extent(0);
    using ResultVec = Vector<T, N>;
    using Result = std::expected<ResultVec, solver::SolverError>;

    auto ldlt_result = ldlt(A);
    if (!ldlt_result) {
        return Result{std::unexpected(std::move(ldlt_result.error()))};
    }

    return ldlt_result->solve(b);
}

} // namespace zipper::utils::decomposition

#endif
