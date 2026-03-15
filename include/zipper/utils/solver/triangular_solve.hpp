/// @file triangular_solve.hpp
/// @brief Forward and back substitution for triangular linear systems.
/// @ingroup solvers
///
/// Solves the system  T * x = b  where T is a triangular matrix presented
/// as a `TriangularView` expression.  The triangular mode (lower/upper,
/// unit/strict/plain diagonal) is known at compile time, enabling efficient
/// dispatch without runtime branching.
///
/// Two interfaces:
///   - `triangular_solve(T, b)` — returns a new vector with the solution.
///   - `triangular_solve_inplace(T, x)` — solves in place (x is both the
///     RHS on entry and the solution on exit).
///
/// Both detect zero pivots and return `SolverError::Kind::breakdown` when
/// a diagonal element is (near-)zero and the mode is not UnitDiag.
///
/// Complexity: O(n^2) for a dense n x n triangular system.
///
/// @see zipper::expression::unary::TriangularView — the expression type that
///      provides the triangular matrix input.
/// @see zipper::expression::TriangularMode — compile-time enum encoding
///      Lower/Upper and diagonal treatment (Unit/Zero/pass-through).
/// @see zipper::utils::solver::SolverResult — result type for iterative solvers
///      (triangular_solve's returning variant uses `std::expected<Vector,...>`
///      directly).
/// @see zipper::utils::decomposition::qr_solve — uses triangular_solve for
///      back-substitution after QR factorisation.
/// @see zipper::utils::solver::gmres — uses triangular_solve for
///      back-substitution on the Hessenberg matrix.
/// @see zipper::utils::detail::inverse_general — uses triangular_solve to
///      solve each column of the inverse via QR + back-substitution.

#pragma once

#include <cmath>
#include <expected>
#include <limits>

#include <zipper/Vector.hpp>
#include <zipper/expression/unary/TriangularView.hpp>
#include <zipper/utils/solver/result.hpp>

namespace zipper::utils::solver {

// ─────────────────────────────────────────────────────────────────────────────
// Implementation details
// ─────────────────────────────────────────────────────────────────────────────
namespace detail {

/// @brief Forward substitution for lower-triangular systems (L * x = b).
///
/// Processes rows from top to bottom.  For row i, the known entries are
/// x[0..i-1], so:
///   x[i] = (b[i] - sum_{j in col_range, j < i} L(i,j) * x[j]) / L(i,i)
///
/// For UnitDiag mode, L(i,i) = 1 so no division is needed.
/// For ZeroDiag (StrictlyLower), the diagonal is 0 and the system is
/// singular — this returns a breakdown error.
template <zipper::expression::TriangularMode Mode, typename TriExpr,
          typename T, zipper::index_type Dim>
auto forward_substitute_inplace(const TriExpr &L,
                                zipper::Vector<T, Dim> &x)
    -> std::expected<void, SolverError> {

    using namespace zipper::expression;

    const auto n = L.extent(0);

    // StrictlyLower has zero diagonal — always singular.
    if constexpr (has_flag(Mode, TriangularMode::ZeroDiag)) {
        return std::unexpected(SolverError{
            .kind = SolverError::Kind::breakdown,
            .message = "Triangular solve: strictly lower-triangular "
                       "matrix has zero diagonal (singular)"});
    } else {
        for (zipper::index_type i = 0; i < n; ++i) {
            auto col_range = L.col_range_for_row(i);

            T sum = x(i);
            for (auto j_it = col_range.begin(); j_it != col_range.end();
                 ++j_it) {
                auto j = static_cast<zipper::index_type>(*j_it);
                if (j < i) {
                    sum -= L.coeff(i, j) * x(j);
                }
            }

            if constexpr (has_flag(Mode, TriangularMode::UnitDiag)) {
                // Diagonal is 1, no division needed.
                x(i) = sum;
            } else {
                T diag = L.coeff(i, i);
                if (std::abs(diag) <=
                    std::numeric_limits<T>::epsilon()) {
                    return std::unexpected(SolverError{
                        .kind = SolverError::Kind::breakdown,
                        .message = "Triangular solve: zero pivot at "
                                   "diagonal index " +
                                   std::to_string(i)});
                }
                x(i) = sum / diag;
            }
        }
        return {};
    }
}

/// @brief Back substitution for upper-triangular systems (U * x = b).
///
/// Processes rows from bottom to top.  For row i, the known entries are
/// x[i+1..n-1], so:
///   x[i] = (b[i] - sum_{j in col_range, j > i} U(i,j) * x[j]) / U(i,i)
template <zipper::expression::TriangularMode Mode, typename TriExpr,
          typename T, zipper::index_type Dim>
auto back_substitute_inplace(const TriExpr &U,
                             zipper::Vector<T, Dim> &x)
    -> std::expected<void, SolverError> {

    using namespace zipper::expression;

    const auto n = U.extent(0);

    // StrictlyUpper has zero diagonal — always singular.
    if constexpr (has_flag(Mode, TriangularMode::ZeroDiag)) {
        return std::unexpected(SolverError{
            .kind = SolverError::Kind::breakdown,
            .message = "Triangular solve: strictly upper-triangular "
                       "matrix has zero diagonal (singular)"});
    } else {
        for (zipper::index_type ii = n; ii > 0; --ii) {
            zipper::index_type i = ii - 1;
            auto col_range = U.col_range_for_row(i);

            T sum = x(i);
            for (auto j_it = col_range.begin(); j_it != col_range.end();
                 ++j_it) {
                auto j = static_cast<zipper::index_type>(*j_it);
                if (j > i) {
                    sum -= U.coeff(i, j) * x(j);
                }
            }

            if constexpr (has_flag(Mode, TriangularMode::UnitDiag)) {
                x(i) = sum;
            } else {
                T diag = U.coeff(i, i);
                if (std::abs(diag) <=
                    std::numeric_limits<T>::epsilon()) {
                    return std::unexpected(SolverError{
                        .kind = SolverError::Kind::breakdown,
                        .message = "Triangular solve: zero pivot at "
                                   "diagonal index " +
                                   std::to_string(i)});
                }
                x(i) = sum / diag;
            }
        }
        return {};
    }
}

}  // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Solves T * x = b in place, where x contains b on entry and the
///        solution on exit.
///
/// @tparam Mode  Compile-time triangular mode of the matrix view.
/// @param  tri   The triangular view of the coefficient matrix.
/// @param  x     On entry: the RHS vector b.  On exit: the solution x.
/// @return `std::expected<void, SolverError>` — empty on success, or a
///         `SolverError` on breakdown (zero pivot / singular matrix).
///
/// @code
///   auto M = Matrix<double, 3, 3>({...});
///   auto L = triangular_view<TriangularMode::Lower>(M);
///   Vector<double, 3> x = b;  // copy b into x
///   auto result = triangular_solve_inplace(L, x);
///   if (result) { /* x now contains the solution */ }
/// @endcode
template <zipper::expression::TriangularMode Mode, typename ExprType,
          typename T, zipper::index_type Dim>
auto triangular_solve_inplace(
    const zipper::expression::unary::TriangularView<Mode, ExprType> &tri,
    zipper::Vector<T, Dim> &x)
    -> std::expected<void, SolverError> {

    using namespace zipper::expression;

    if constexpr (has_flag(Mode, TriangularMode::Lower)) {
        return detail::forward_substitute_inplace<Mode>(tri, x);
    } else {
        return detail::back_substitute_inplace<Mode>(tri, x);
    }
}

/// @brief Solves T * x = b, returning the solution as a new vector.
///
/// @tparam Mode  Compile-time triangular mode (deduced from tri).
/// @param  tri   The triangular view of the coefficient matrix.
/// @param  b     The right-hand side vector.
/// @return `std::expected<Vector<T, Dim>, SolverError>` — the solution on
///         success, or a `SolverError` on breakdown.
///
/// @code
///   auto M = Matrix<double, 3, 3>({...});
///   auto L = triangular_view<TriangularMode::Lower>(M);
///   auto result = triangular_solve(L, b);
///   if (result) { auto x = result->x; }
/// @endcode
template <zipper::expression::TriangularMode Mode, typename ExprType,
          zipper::concepts::Vector BDerived>
auto triangular_solve(
    const zipper::expression::unary::TriangularView<Mode, ExprType> &tri,
    const BDerived &b) {
    using T = typename BDerived::value_type;
    constexpr auto Dim = BDerived::extents_type::static_extent(0);

    using ResultVec = zipper::Vector<T, Dim>;
    using Result = std::expected<ResultVec, SolverError>;

    // Copy b into x (the solution vector we'll modify in place).
    ResultVec x(b);

    auto status = triangular_solve_inplace(tri, x);
    if (!status) {
        return Result{std::unexpected(std::move(status.error()))};
    }
    return Result{std::move(x)};
}

}  // namespace zipper::utils::solver
