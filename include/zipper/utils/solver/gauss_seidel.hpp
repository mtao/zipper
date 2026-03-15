/// @file gauss_seidel.hpp
/// @brief Gauss-Seidel iterative method for solving the linear system Ax = b.
/// @ingroup solvers
///
/// The Gauss-Seidel method improves upon Jacobi by using the most recently
/// computed values of x as soon as they are available.  The update rule is:
///
///     x_i  <-  x_i + (b_i - dot(A_row_i, x)) / A_{ii}
///
/// This corresponds to a forward sweep through the rows, and the method is
/// equivalent to solving  (D + L) * x^{k+1} = b - U * x^k, where D is the
/// diagonal, L the strict lower triangle, and U the strict upper triangle of A.
///
/// Convergence is guaranteed when A is:
///   - Strictly diagonally dominant, or
///   - Symmetric positive definite.
///
/// Gauss-Seidel generally converges faster than Jacobi (roughly twice as fast
/// in terms of iteration count for many problems) because updated components
/// propagate information within the same iteration.  However, unlike Jacobi,
/// the updates are inherently sequential -- component i depends on the already-
/// updated components 0..i-1 from the current iteration.
///
/// Two overloads are provided:
///   - `gauss_seidel(A, b, x0, tol, max_iter)` -- with initial guess.
///   - `gauss_seidel(A, b, tol, max_iter)` -- zero initial guess.
///
/// @see zipper::utils::solver::conjugate_gradient — Krylov solver for SPD
///      matrices (faster convergence than Gauss-Seidel for well-conditioned
///      problems, guaranteed convergence in at most n steps).
/// @see zipper::utils::solver::gmres — Krylov solver for general non-symmetric
///      systems.
/// @see zipper::utils::solver::bicgstab — Krylov solver for general
///      non-symmetric systems (two matvecs per iteration).
/// @see zipper::utils::solver::triangular_solve — direct triangular solver
///      (O(n^2)) for triangular systems.
/// @see zipper::utils::solver::SolverResult — the result type returned on
///      convergence.
/// @see zipper::utils::solver::SolverError — the error type returned on
///      failure.

#if !defined(ZIPPER_UTILS_SOLVER_GAUSS_SEIDEL_HPP)
#define ZIPPER_UTILS_SOLVER_GAUSS_SEIDEL_HPP

#include <cmath>
#include <limits>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Constant.hpp>

#include "result.hpp"

namespace zipper::utils::solver {

/// @brief Gauss-Seidel iteration with an explicit initial guess.
///
/// @param A        Square coefficient matrix.
/// @param b        Right-hand side vector.
/// @param x0       Initial guess for the solution.
/// @param tol      Convergence tolerance on ||b - A*x||_2 (default: 1e-10).
/// @param max_iter Maximum number of iterations (default: 1000).
///
/// @return On success, a `SolverResult` with the solution, residual norm, and
///         iteration count.  On failure, a `SolverError`.
///
/// A zero diagonal element causes immediate failure with a `breakdown` error,
/// since division by zero would produce nonsense.
template <concepts::Matrix ADerived, concepts::Vector BDerived,
          concepts::Vector XDerived>
auto gauss_seidel(const ADerived &A, const BDerived &b, const XDerived &x0,
                  typename std::decay_t<ADerived>::value_type tol = 1e-10,
                  index_type max_iter = 1000) {
  using AType = std::decay_t<ADerived>;
  using T = typename AType::value_type;
  constexpr index_type Dim = AType::extents_type::static_extent(0);
  using Vec = Vector<T, Dim>;
  using Result = SolverResult<T, Dim>;

  const index_type n = A.extent(0);

  Vec x(x0);

  for (index_type iter = 0; iter < max_iter; ++iter) {
    // Forward sweep: update each component using the latest values.
    for (index_type i = 0; i < n; ++i) {
      T diag = A(i, i);

      if (std::abs(diag) < std::numeric_limits<T>::min()) {
        return std::expected<Result, SolverError>{std::unexpected(SolverError{
            .kind = SolverError::Kind::breakdown,
            .message = "Gauss-Seidel breakdown: zero diagonal element",
        })};
      }

      T dot = T{0};
      for (index_type j = 0; j < n; ++j) {
        dot += A(i, j) * x(j);
      }
      x(i) += (b(i) - dot) / diag;
    }

    // Check convergence.
    Vec r(b - A * x);
    T rnorm = r.norm();

    if (rnorm <= tol) {
      return std::expected<Result, SolverError>{
          Result{.x = std::move(x), .residual_norm = rnorm, .iterations = iter + 1}};
    }

    if (rnorm > T{1e30}) {
      return std::expected<Result, SolverError>{std::unexpected(SolverError{
          .kind = SolverError::Kind::diverged,
          .message =
              "Gauss-Seidel iteration diverged: residual norm exceeded 1e30",
      })};
    }
  }

  return std::expected<Result, SolverError>{std::unexpected(SolverError{
      .kind = SolverError::Kind::diverged,
      .message =
          "Gauss-Seidel iteration did not converge within max_iter iterations",
  })};
}

/// @brief Gauss-Seidel iteration starting from the zero vector.
///
/// Convenience overload -- see the other overload for full documentation.
template <concepts::Matrix ADerived, concepts::Vector BDerived>
auto gauss_seidel(const ADerived &A, const BDerived &b,
                  typename std::decay_t<ADerived>::value_type tol = 1e-10,
                  index_type max_iter = 1000) {
  using AType = std::decay_t<ADerived>;
  using T = typename AType::value_type;
  constexpr index_type Dim = AType::extents_type::static_extent(0);

  const index_type n = A.extent(0);
  Vector<T, Dim> x0(n);
  x0 = expression::nullary::Constant(T{0}, x0.extents());

  return gauss_seidel(A, b, x0, tol, max_iter);
}

} // namespace zipper::utils::solver

#endif
