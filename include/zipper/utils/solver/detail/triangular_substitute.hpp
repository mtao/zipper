/// @file triangular_substitute.hpp
/// @brief Forward and back substitution implementation detail.
///
/// These functions implement the core substitution loops for triangular
/// systems.  They are factored out so that TriangularView::solve() can
/// use them without creating a circular include dependency.
///
/// @see zipper::expression::unary::TriangularView::solve

#if !defined(ZIPPER_UTILS_SOLVER_DETAIL_TRIANGULAR_SUBSTITUTE_HPP)
#define ZIPPER_UTILS_SOLVER_DETAIL_TRIANGULAR_SUBSTITUTE_HPP

#include <cmath>
#include <expected>
#include <limits>
#include <string>

#include <zipper/Vector.hpp>
#include <zipper/expression/TriangularMode.hpp>
#include <zipper/utils/solver/result.hpp>

namespace zipper::utils::solver::detail {

/// @brief Forward substitution for lower-triangular systems (L * x = b).
///
/// Processes rows from top to bottom.  For row i, the known entries are
/// x[0..i-1], so:
///   x[i] = (b[i] - sum_{j in col_range, j < i} L(i,j) * x[j]) / L(i,i)
///
/// For UnitDiag mode, L(i,i) = 1 so no division is needed.
/// For ZeroDiag (StrictlyLower), the diagonal is 0 and the system is
/// singular -- this returns a breakdown error.
template <zipper::expression::TriangularMode Mode, typename TriExpr,
          typename T, zipper::index_type Dim>
auto forward_substitute_inplace(const TriExpr &L,
                                zipper::Vector<T, Dim> &x)
    -> std::expected<void, SolverError> {

    using namespace zipper::expression;

    const auto n = L.extent(0);

    // StrictlyLower has zero diagonal -- always singular.
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

    // StrictlyUpper has zero diagonal -- always singular.
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

} // namespace zipper::utils::solver::detail
#endif
