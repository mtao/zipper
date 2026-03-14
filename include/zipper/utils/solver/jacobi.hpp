/// @file jacobi.hpp
/// @brief Jacobi iterative method for solving the linear system Ax = b.
///
/// The Jacobi method is a simple stationary iterative solver that decomposes
/// the coefficient matrix A into its diagonal D and off-diagonal remainder
/// R = A - D, then iterates:
///
///     x^{k+1}_i = (1 / A_{ii}) * (b_i - sum_{j != i} A_{ij} * x^k_j)
///
/// This is equivalent to solving  D * x^{k+1} = b - R * x^k  at each step.
///
/// Convergence is guaranteed when A is strictly diagonally dominant (i.e.
/// |A_{ii}| > sum_{j != i} |A_{ij}| for every row i).  It also converges for
/// many symmetric positive definite systems, though Gauss-Seidel or CG will
/// typically be faster for those.
///
/// The method is inherently parallel: every component of x^{k+1} depends only
/// on the *previous* iterate x^k, making it well-suited to GPU or SIMD
/// execution (though this implementation is serial for simplicity).
///
/// Two overloads are provided:
///   - `jacobi(A, b, x0, tol, max_iter)` -- with an explicit initial guess.
///   - `jacobi(A, b, tol, max_iter)` -- uses the zero vector as initial guess.
///
/// Both return `std::expected<SolverResult<T, Dim>, SolverError>`.

#if !defined(ZIPPER_UTILS_SOLVER_JACOBI_HPP)
#define ZIPPER_UTILS_SOLVER_JACOBI_HPP

#include <cmath>
#include <limits>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Constant.hpp>

#include "result.hpp"

namespace zipper::utils::solver {

/// @brief Jacobi iteration with an explicit initial guess.
///
/// @param A        Square coefficient matrix (must be diagonally dominant for
///                 guaranteed convergence).
/// @param b        Right-hand side vector.
/// @param x0       Initial guess for the solution.
/// @param tol      Convergence tolerance on the 2-norm of the residual
///                 (default: 1e-10).
/// @param max_iter Maximum number of iterations (default: 1000).
///
/// @return On success, a `SolverResult` containing the approximate solution,
///         the final residual norm, and the iteration count.  On failure, a
///         `SolverError` describing why the solver did not converge.
///
/// The iteration terminates when ||b - A*x||_2 <= tol, or after max_iter
/// iterations.  If the residual norm ever exceeds 1e30 the solver reports
/// divergence immediately rather than wasting further iterations.
template <concepts::Matrix ADerived, concepts::Vector BDerived,
          concepts::Vector XDerived>
auto jacobi(const ADerived &A, const BDerived &b, const XDerived &x0,
            typename std::decay_t<ADerived>::value_type tol = 1e-10,
            index_type max_iter = 1000) {
  using AType = std::decay_t<ADerived>;
  using T = typename AType::value_type;
  constexpr index_type Dim = AType::extents_type::static_extent(0);
  using Vec = Vector<T, Dim>;
  using Result = SolverResult<T, Dim>;

  const index_type n = A.extent(0);

  Vec x(x0);
  Vec x_new(x0);

  for (index_type iter = 0; iter < max_iter; ++iter) {
    // Compute x_new component-by-component.  Each component of x_new
    // depends only on the *previous* iterate x (not on other components
    // of x_new), which is what distinguishes Jacobi from Gauss-Seidel.
    for (index_type i = 0; i < n; ++i) {
      T sigma = T{0};
      for (index_type j = 0; j < n; ++j) {
        if (j != i) {
          sigma += A(i, j) * x(j);
        }
      }
      x_new(i) = (b(i) - sigma) / A(i, i);
    }

    x = x_new;

    // Check convergence: compute residual r = b - A*x.
    Vec r(b - A * x);
    T rnorm = r.norm();

    if (rnorm <= tol) {
      return std::expected<Result, SolverError>{
          Result{.x = std::move(x), .residual_norm = rnorm, .iterations = iter + 1}};
    }

    if (rnorm > T{1e30}) {
      return std::expected<Result, SolverError>{std::unexpected(SolverError{
          .kind = SolverError::Kind::diverged,
          .message = "Jacobi iteration diverged: residual norm exceeded 1e30",
      })};
    }
  }

  // Did not converge within max_iter iterations.
  return std::expected<Result, SolverError>{std::unexpected(SolverError{
      .kind = SolverError::Kind::diverged,
      .message = "Jacobi iteration did not converge within max_iter iterations",
  })};
}

/// @brief Jacobi iteration starting from the zero vector.
///
/// Convenience overload that constructs a zero initial guess and delegates to
/// the full version.  See the other overload for parameter documentation.
template <concepts::Matrix ADerived, concepts::Vector BDerived>
auto jacobi(const ADerived &A, const BDerived &b,
            typename std::decay_t<ADerived>::value_type tol = 1e-10,
            index_type max_iter = 1000) {
  using AType = std::decay_t<ADerived>;
  using T = typename AType::value_type;
  constexpr index_type Dim = AType::extents_type::static_extent(0);

  const index_type n = A.extent(0);
  Vector<T, Dim> x0(n);
  x0 = expression::nullary::Constant(T{0}, x0.extents());

  return jacobi(A, b, x0, tol, max_iter);
}

} // namespace zipper::utils::solver

#endif
