/// @file conjugate_gradient.hpp
/// @brief Conjugate Gradient (CG) method for symmetric positive definite
///        linear systems Ax = b.
///
/// The Conjugate Gradient method is the gold-standard Krylov solver for
/// symmetric positive definite (SPD) matrices.  It maintains a residual vector
/// r, a search direction p, and updates the solution x along A-conjugate
/// directions so that the error is minimised in the A-norm at each step.
///
/// The algorithm:
///   r_0 = b - A * x_0
///   p_0 = r_0
///   for k = 0, 1, 2, ...
///     alpha_k   = (r_k^T r_k) / (p_k^T A p_k)
///     x_{k+1}   = x_k + alpha_k * p_k
///     r_{k+1}   = r_k - alpha_k * A * p_k
///     beta_{k+1} = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
///     p_{k+1}   = r_{k+1} + beta_{k+1} * p_k
///
/// For an n x n SPD matrix, CG is guaranteed to converge in at most n
/// iterations in exact arithmetic.  In floating-point it typically converges
/// much faster, especially when the eigenvalues of A are clustered.
///
/// CG requires only one matrix-vector product per iteration and a handful of
/// dot products and vector updates, making it very efficient for large sparse
/// systems (though zipper currently stores dense matrices).
///
/// Two overloads:
///   - `conjugate_gradient(A, b, x0, tol, max_iter)` -- with initial guess.
///   - `conjugate_gradient(A, b, tol, max_iter)` -- zero initial guess.

#if !defined(ZIPPER_UTILS_SOLVER_CONJUGATE_GRADIENT_HPP)
#define ZIPPER_UTILS_SOLVER_CONJUGATE_GRADIENT_HPP

#include <cmath>
#include <limits>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/utils/detail/dot.hpp>

#include "result.hpp"

namespace zipper::utils::solver {

/// @brief Conjugate Gradient with an explicit initial guess.
///
/// @param A        Symmetric positive definite coefficient matrix.
/// @param b        Right-hand side vector.
/// @param x0       Initial guess for the solution.
/// @param tol      Convergence tolerance on ||r||_2 (default: 1e-10).
/// @param max_iter Maximum number of iterations (default: 1000).
///
/// @return `SolverResult` on convergence, `SolverError` on failure.
///
/// The solver detects breakdown when p^T A p is too small (would cause
/// division by near-zero), which can happen if A is not actually SPD.
template <concepts::Matrix ADerived, concepts::Vector BDerived,
          concepts::Vector XDerived>
auto conjugate_gradient(const ADerived &A, const BDerived &b,
                        const XDerived &x0,
                        typename std::decay_t<ADerived>::value_type tol = 1e-10,
                        index_type max_iter = 1000) {
  using AType = std::decay_t<ADerived>;
  using T = typename AType::value_type;
  constexpr index_type Dim = AType::extents_type::static_extent(0);
  using Vec = Vector<T, Dim>;
  using Result = SolverResult<T, Dim>;

  Vec x(x0);
  Vec r(b - A * x);
  Vec p(r);

  T r_dot_r = utils::detail::dot(r, r);

  for (index_type iter = 0; iter < max_iter; ++iter) {
    T rnorm = std::sqrt(r_dot_r);

    if (rnorm <= tol) {
      return std::expected<Result, SolverError>{
          Result{.x = std::move(x), .residual_norm = rnorm, .iterations = iter}};
    }

    // A * p_k
    Vec Ap(A * p);

    T pAp = utils::detail::dot(p, Ap);

    if (std::abs(pAp) < std::numeric_limits<T>::min()) {
      return std::expected<Result, SolverError>{std::unexpected(SolverError{
          .kind = SolverError::Kind::breakdown,
          .message = "CG breakdown: p^T A p is near zero (A may not be SPD)",
      })};
    }

    T alpha = r_dot_r / pAp;

    // x_{k+1} = x_k + alpha * p_k
    x += alpha * p;

    // r_{k+1} = r_k - alpha * A * p_k
    r -= alpha * Ap;

    T r_dot_r_new = utils::detail::dot(r, r);
    T beta = r_dot_r_new / r_dot_r;

    // p_{k+1} = r_{k+1} + beta * p_k
    // Aliasing note: Vec(...) fully evaluates into new storage before
    // assignment, so this is safe even though p appears on both sides.
    p = Vec(r + beta * p);

    r_dot_r = r_dot_r_new;
  }

  return std::expected<Result, SolverError>{std::unexpected(SolverError{
      .kind = SolverError::Kind::diverged,
      .message = "CG did not converge within max_iter iterations",
  })};
}

/// @brief Conjugate Gradient starting from the zero vector.
///
/// Convenience overload -- see the other overload for full documentation.
template <concepts::Matrix ADerived, concepts::Vector BDerived>
auto conjugate_gradient(const ADerived &A, const BDerived &b,
                        typename std::decay_t<ADerived>::value_type tol = 1e-10,
                        index_type max_iter = 1000) {
  using AType = std::decay_t<ADerived>;
  using T = typename AType::value_type;
  constexpr index_type Dim = AType::extents_type::static_extent(0);

  const index_type n = A.extent(0);
  Vector<T, Dim> x0(n);
  x0 = expression::nullary::Constant(T{0}, x0.extents());

  return conjugate_gradient(A, b, x0, tol, max_iter);
}

} // namespace zipper::utils::solver

#endif
