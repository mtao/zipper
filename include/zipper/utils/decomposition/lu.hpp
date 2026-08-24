/// @file lu.hpp
/// @brief PLU decomposition (LU with partial pivoting) for general square
///        matrices.
///
/// Given a square matrix A, the PLU decomposition factors it as:
///
///   P * A = L * U
///
/// where P is a permutation matrix, L is a unit lower triangular matrix (ones
/// on the diagonal), and U is an upper triangular matrix.  The permutation P
/// encodes the row swaps performed during Gaussian elimination with partial
/// pivoting (selecting the largest-magnitude pivot element in each column).
///
/// The factored form stores L and U in a single combined matrix: the strict
/// lower triangle holds the multipliers of L (with implicit unit diagonal),
/// and the upper triangle (including the diagonal) holds U.  This matches the
/// standard LAPACK convention.
///
/// The permutation is stored as a DataArray `perm` where `perm(i)` is the
/// original row index of the row currently in position i after all swaps.
/// A sign integer tracks the parity of the permutation (for computing
/// determinants).
///
/// Three interfaces:
///   - `plu(A)`          -- returns the PLU factorisation.
///   - `PLUResult::solve(b)` -- solves Ax = b using the stored factorisation.
///   - `plu_solve(A, b)` -- convenience: factorise then solve in one call.
///
/// The solve proceeds in three stages:
///   1. Permute:             Pb = P * b
///   2. Forward substitution: L * y = Pb   (unit lower triangular)
///   3. Back substitution:    U * x = y    (upper triangular)
///
/// Singularity is detected when a zero (or near-zero) pivot is encountered
/// during factorisation.  The functions return `std::expected` with a
/// `SolverError` on failure.
///
/// Complexity: O(2n^3/3) for the factorisation, O(n^2) for each solve.

#if !defined(ZIPPER_UTILS_DECOMPOSITION_LU_HPP)
#define ZIPPER_UTILS_DECOMPOSITION_LU_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <expected>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

#include <zipper/Array.hpp>
#include <zipper/DataArray.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Iota.hpp>
#include <zipper/expression/unary/TriangularView.hpp>
#include <zipper/utils/decomposition/detail/shape_validation.hpp>
#include <zipper/utils/max_coeff.hpp>
#include <zipper/utils/solver/result.hpp>

namespace zipper::utils::decomposition {

// ─────────────────────────────────────────────────────────────────────────────
// Result type
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a PLU decomposition.
///
/// `LU` is an n x n matrix whose lower triangle stores the multipliers of L
/// (with an implicit unit diagonal) and whose upper triangle (including the
/// diagonal) stores U.  `perm` is the row permutation vector.  `sign` is +1
/// or -1 depending on the parity of the permutation (useful for determinants).
///
/// Calling `.solve(b)` performs permuted forward/back substitution using the
/// stored factors to solve A*x = b without re-factoring.
template <typename T, index_type N> struct PLUResult {
    /// Scalar type of the decomposition.
    using value_type = T;

    /// Combined LU matrix (lower triangle = L multipliers, upper triangle = U).
    Matrix<T, N, N> LU;
    /// Row permutation: perm[i] = original row index at position i.
    DataArray<index_type, N> perm;
    /// Sign of the permutation (+1 or -1).
    int sign;

    /// @brief Unit lower triangular factor L.
    ///
    /// Returns a read-only TriangularView of the combined LU matrix with
    /// UnitLower mode: the strict lower triangle contains the L multipliers,
    /// and the diagonal is implicitly 1.
    auto L() const & {
        return LU.template as_triangular<expression::TriangularMode::UnitLower>();
    }
    auto L() const && = delete;

    /// @brief Upper triangular factor U.
    ///
    /// Returns a read-only TriangularView of the combined LU matrix with
    /// Upper mode: the upper triangle (including the diagonal) contains U.
    auto U() const & {
        return LU.template as_triangular<expression::TriangularMode::Upper>();
    }
    auto U() const && = delete;

    /// @brief Solve A*x = b using the stored PLU factors.
    ///
    /// Performs:
    ///   1. Apply permutation to b:  Pb(i) = b(perm(i))
    ///   2. Forward substitution:    L * y = Pb  (unit lower triangular)
    ///   3. Back substitution:       U * x = y   (upper triangular)
    ///
    /// @param b  Right-hand side vector of length n.
    /// @return   `std::expected<Vector<T,N>, SolverError>` — the solution on
    ///           success, or a breakdown error if a zero pivot is encountered.
    template <concepts::Vector BDerived>
        requires detail::StaticallyCompatibleRhs<decltype(LU), BDerived>
    auto solve(const BDerived &b) const
        -> std::expected<Vector<T, N>, solver::SolverError> {
        using ResultVec = Vector<T, N>;
        using Result = std::expected<ResultVec, solver::SolverError>;

        if (auto valid = detail::validate_square(LU, "PLU solve"); !valid) {
            return Result{std::unexpected(std::move(valid.error()))};
        }
        if (perm.extent(0) != LU.extent(0)) {
            return Result{std::unexpected(detail::invalid_shape(
                "PLU solve: permutation size must match rows"))};
        }
        if (auto valid = detail::validate_rhs(LU.extent(0), b, "PLU solve");
            !valid) {
            return Result{std::unexpected(std::move(valid.error()))};
        }

        // 1. Apply permutation: Pb(i) = b(perm(i)).
        ResultVec Pb(b(perm));

        // 2. Forward substitution: L * y = Pb  (L is unit lower triangular).
        auto y_result = L().solve(Pb);

        if (!y_result) {
            return Result{std::unexpected(std::move(y_result.error()))};
        }

        // 3. Back substitution: U * x = y  (U is upper triangular).
        auto x_result = U().solve(*y_result);

        if (!x_result) {
            return Result{std::unexpected(std::move(x_result.error()))};
        }

        return Result{std::move(*x_result)};
    }

    /// @brief Compute the determinant of the original matrix A.
    ///
    /// det(A) = sign * product of U diagonal entries.
    auto determinant() const -> T {
        return static_cast<T>(sign) * LU.diagonal().as_array().product();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// PLU factorisation
// ─────────────────────────────────────────────────────────────────────────────

/// @brief PLU decomposition of a general square matrix.
///
/// Performs Gaussian elimination with partial pivoting (largest-magnitude
/// pivot in each column).
///
/// @param A  An n x n matrix.
/// @return   `std::expected<PLUResult<T,N>, SolverError>` — the factors on
///           success, or a breakdown error if the matrix is singular (zero
///           pivot encountered).
template <concepts::Matrix Derived>
    requires detail::StaticallySquare<Derived>
auto plu(const Derived &A)
    -> std::expected<
        PLUResult<typename std::decay_t<Derived>::value_type,
                  std::decay_t<Derived>::extents_type::static_extent(0)>,
        solver::SolverError> {
    using AType = std::decay_t<Derived>;
    using T = typename AType::value_type;
    constexpr index_type N = AType::extents_type::static_extent(0);
    using Result = std::expected<PLUResult<T, N>, solver::SolverError>;

    if (auto valid = detail::validate_square(A, "PLU decomposition"); !valid) {
        return Result{std::unexpected(std::move(valid.error()))};
    }

    const index_type n = A.rows();

    // Copy A into the combined LU matrix via converting constructor.
    Matrix<T, N, N> LU(A);

    // Initialise the permutation vector to the identity.
    DataArray<index_type, N> perm(
        expression::nullary::iota<index_type>(index_type{0},
                                               zipper::extents<N>(n)));
    int sign = 1;

    for (index_type k = 0; k < n; ++k) {
        // Partial pivoting: find the row with the largest absolute value in
        // column k (from row k downward).
        const auto [max_val, max_index] = utils::maxCoeffWithIndex(
            LU.col(k).segment(k, n - k).as_array().abs());
        const index_type max_row = k + max_index[0];

        // Check for singular matrix.
        if (max_val <= std::numeric_limits<T>::epsilon()) {
            return Result{std::unexpected(solver::SolverError{
                .kind = solver::SolverError::Kind::breakdown,
                .message = "PLU decomposition: matrix is singular (zero pivot at column " +
                           std::to_string(k) + ")"})};
        }

        // Swap rows if necessary.
        if (max_row != k) {
            auto rows = LU.row_slice(
                std::array<index_type, 2>{k, max_row});
            rows = rows.row_slice(std::array<index_type, 2>{1, 0}).eval();
            std::swap(perm(k), perm(max_row));
            sign = -sign;
        }

        // Gaussian elimination: compute multipliers and update trailing submatrix.
        const T pivot = LU(k, k);
        if (k + 1 < n) {
            auto multipliers = LU.col(k).segment(k + 1, n - k - 1);
            multipliers /= pivot;

            auto trailing = LU.slice(zipper::slice(k + 1, n - k - 1),
                                     zipper::slice(k + 1, n - k - 1));
            auto pivot_row = LU.row(k).segment(k + 1, n - k - 1);
            auto outer_product = multipliers.lift().as_array() *
                                 pivot_row.lift().transpose().as_array();
            auto trailing_array = trailing.as_array();
            trailing_array = (trailing_array - outer_product).eval();
        }
    }

    return Result{PLUResult<T, N>{
        .LU = std::move(LU), .perm = std::move(perm), .sign = sign}};
}

// ─────────────────────────────────────────────────────────────────────────────
// PLU solve
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Solve Ax = b via PLU decomposition.
///
/// Factors P*A = L*U, then delegates to `PLUResult::solve(b)`.
///
/// @param A  An n x n matrix.
/// @param b  Right-hand side vector of length n.
/// @return   `std::expected<Vector<T,N>, SolverError>` — the solution on
///           success, or a breakdown error if the matrix is singular.
template <concepts::Matrix ADerived, concepts::Vector BDerived>
    requires detail::StaticallySquare<ADerived> &&
             detail::StaticallyCompatibleRhs<ADerived, BDerived>
auto plu_solve(const ADerived &A, const BDerived &b)
    -> std::expected<
        Vector<typename std::decay_t<ADerived>::value_type,
               std::decay_t<ADerived>::extents_type::static_extent(0)>,
        solver::SolverError> {
    using AType = std::decay_t<ADerived>;
    using T = typename AType::value_type;
    constexpr index_type N = AType::extents_type::static_extent(0);
    using ResultVec = Vector<T, N>;
    using Result = std::expected<ResultVec, solver::SolverError>;

    if (auto valid = detail::validate_rhs(A.extent(0), b, "PLU solve");
        !valid) {
        return Result{std::unexpected(std::move(valid.error()))};
    }

    auto plu_result = plu(A);
    if (!plu_result) {
        return Result{std::unexpected(std::move(plu_result.error()))};
    }

    return plu_result->solve(b);
}

} // namespace zipper::utils::decomposition

#endif
