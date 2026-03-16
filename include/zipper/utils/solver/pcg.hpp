/// @file pcg.hpp
/// @brief Preconditioned Conjugate Gradient (PCG) solver for symmetric
///        positive definite linear systems Ax = b.
/// @ingroup solvers
///
/// The Preconditioned Conjugate Gradient method enhances the standard CG
/// algorithm by incorporating a preconditioner M that approximates A.
/// Instead of minimising the A-norm of the error directly, PCG minimises
/// it in the M^{-1}A-norm, which clusters eigenvalues and accelerates
/// convergence.
///
/// The algorithm:
///   r_0 = b - A * x_0
///   z_0 = M^{-1} * r_0
///   p_0 = z_0
///   for k = 0, 1, 2, ...
///     alpha_k   = (r_k^T z_k) / (p_k^T A p_k)
///     x_{k+1}   = x_k + alpha_k * p_k
///     r_{k+1}   = r_k - alpha_k * A * p_k
///     z_{k+1}   = M^{-1} * r_{k+1}
///     beta_{k+1} = (r_{k+1}^T z_{k+1}) / (r_k^T z_k)
///     p_{k+1}   = z_{k+1} + beta_{k+1} * p_k
///
/// When M = I (identity), this reduces to standard CG.
///
/// Two overloads:
///   - `preconditioned_conjugate_gradient(A, b, precond, x0, tol, max_iter)`
///     -- with initial guess.
///   - `preconditioned_conjugate_gradient(A, b, precond, tol, max_iter)`
///     -- zero initial guess.
///
/// @see zipper::utils::solver::conjugate_gradient — unpreconditioned CG.
/// @see zipper::utils::solver::JacobiPreconditioner — diagonal preconditioner.
/// @see zipper::utils::solver::SSORPreconditioner — SSOR preconditioner.
/// @see zipper::concepts::Preconditioner — concept for preconditioners.

#if !defined(ZIPPER_UTILS_SOLVER_PCG_HPP)
#define ZIPPER_UTILS_SOLVER_PCG_HPP

#include <cmath>
#include <limits>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/concepts/Preconditioner.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/utils/detail/dot.hpp>

#include "result.hpp"

namespace zipper::utils::solver {

/// @brief Preconditioned Conjugate Gradient with an explicit initial guess.
///
/// @param A        Symmetric positive definite coefficient matrix.
/// @param b        Right-hand side vector.
/// @param precond  Preconditioner satisfying `concepts::Preconditioner`.
/// @param x0       Initial guess for the solution.
/// @param tol      Convergence tolerance on ||r||_2 (default: 1e-10).
/// @param max_iter Maximum number of iterations (default: 1000).
///
/// @return `SolverResult` on convergence, `SolverError` on failure.
template <concepts::Matrix ADerived, concepts::Vector BDerived,
          concepts::Preconditioner PDerived, concepts::Vector XDerived>
auto preconditioned_conjugate_gradient(
    const ADerived &A, const BDerived &b, const PDerived &precond,
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
  Vec z(precond.apply(r));
  Vec p(z);

  T r_dot_z = utils::detail::dot(r, z);

  for (index_type iter = 0; iter < max_iter; ++iter) {
    T rnorm = std::sqrt(utils::detail::dot(r, r));

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
          .message =
              "PCG breakdown: p^T A p is near zero (A may not be SPD)",
      })};
    }

    T alpha = r_dot_z / pAp;

    // x_{k+1} = x_k + alpha * p_k
    x += alpha * p;

    // r_{k+1} = r_k - alpha * A * p_k
    r -= alpha * Ap;

    // z_{k+1} = M^{-1} * r_{k+1}
    z = Vec(precond.apply(r));

    T r_dot_z_new = utils::detail::dot(r, z);
    T beta = r_dot_z_new / r_dot_z;

    // p_{k+1} = z_{k+1} + beta * p_k
    // Aliasing note: Vec(...) fully evaluates into new storage before
    // assignment, so this is safe even though p appears on both sides.
    p = Vec(z + beta * p);

    r_dot_z = r_dot_z_new;
  }

  return std::expected<Result, SolverError>{std::unexpected(SolverError{
      .kind = SolverError::Kind::diverged,
      .message = "PCG did not converge within max_iter iterations",
  })};
}

/// @brief Preconditioned Conjugate Gradient starting from the zero vector.
///
/// Convenience overload -- see the other overload for full documentation.
template <concepts::Matrix ADerived, concepts::Vector BDerived,
          concepts::Preconditioner PDerived>
auto preconditioned_conjugate_gradient(
    const ADerived &A, const BDerived &b, const PDerived &precond,
    typename std::decay_t<ADerived>::value_type tol = 1e-10,
    index_type max_iter = 1000) {
  using AType = std::decay_t<ADerived>;
  using T = typename AType::value_type;
  constexpr index_type Dim = AType::extents_type::static_extent(0);

  const index_type n = A.extent(0);
  Vector<T, Dim> x0(n);
  x0 = expression::nullary::Constant(T{0}, x0.extents());

  return preconditioned_conjugate_gradient(A, b, precond, x0, tol, max_iter);
}

} // namespace zipper::utils::solver

#endif
