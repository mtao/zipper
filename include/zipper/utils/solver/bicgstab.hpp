/// @file bicgstab.hpp
/// @brief Bi-Conjugate Gradient Stabilized (BiCGSTAB) method for general
///        non-symmetric linear systems Ax = b.
/// @ingroup solvers
///
/// BiCGSTAB is a Krylov subspace method for solving non-symmetric (and
/// possibly non-positive-definite) linear systems.  It was developed by
/// H.A. van der Vorst (1992) as a smoother-converging alternative to the
/// Bi-Conjugate Gradient (BiCG) method, which can exhibit erratic convergence
/// behaviour.
///
/// The algorithm maintains two sequences of vectors — the residual r and a
/// "shadow" residual r_hat (kept fixed as r_0) — and combines a BiCG-like
/// step with a minimisation step (GMRES(1)-like) to stabilise convergence.
///
/// Algorithm:
///   r_0 = b - A * x_0
///   r_hat = r_0           (shadow residual, never updated)
///   p_0 = r_0
///   for k = 0, 1, 2, ...
///     alpha   = (r_hat^T r_k) / (r_hat^T A p_k)
///     s       = r_k - alpha * A * p_k
///     omega   = (A s)^T s / (A s)^T (A s)
///     x_{k+1} = x_k + alpha * p_k + omega * s
///     r_{k+1} = s - omega * A * s
///     beta    = (r_hat^T r_{k+1}) / (r_hat^T r_k) * (alpha / omega)
///     p_{k+1} = r_{k+1} + beta * (p_k - omega * A * p_k)
///
/// BiCGSTAB requires two matrix-vector products per iteration but avoids the
/// need to transpose A (unlike BiCG), making it practical for operators that
/// are only available as "black-box" matvecs.
///
/// The method can break down if r_hat^T r_k becomes zero (the shadow residual
/// becomes orthogonal to the current residual) or if omega becomes zero.
/// These conditions are detected and reported as `SolverError::Kind::breakdown`.
///
/// Two overloads:
///   - `bicgstab(A, b, x0, tol, max_iter)` -- with initial guess.
///   - `bicgstab(A, b, tol, max_iter)` -- zero initial guess.
///
/// @see zipper::utils::solver::gmres — alternative Krylov solver for general
///      non-symmetric systems (one matvec per iteration but growing storage).
/// @see zipper::utils::solver::conjugate_gradient — Krylov solver restricted
///      to SPD matrices (one matvec per iteration, guaranteed convergence).
/// @see zipper::utils::solver::gauss_seidel — stationary iterative method
///      for diagonally-dominant or SPD systems.
/// @see zipper::expression::unary::TriangularView::solve — direct triangular
///      solver (O(n^2)) for triangular systems.
/// @see zipper::utils::solver::SolverResult — the result type returned on
///      convergence.
/// @see zipper::utils::solver::SolverError — the error type returned on
///      failure.

#if !defined(ZIPPER_UTILS_SOLVER_BICGSTAB_HPP)
#define ZIPPER_UTILS_SOLVER_BICGSTAB_HPP

#include <cmath>
#include <limits>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/utils/detail/dot.hpp>

#include "result.hpp"

namespace zipper::utils::solver {

/// @brief BiCGSTAB with an explicit initial guess.
///
/// @param A        Square coefficient matrix (general, non-symmetric OK).
/// @param b        Right-hand side vector.
/// @param x0       Initial guess for the solution.
/// @param tol      Convergence tolerance on ||b - A*x||_2 (default: 1e-10).
/// @param max_iter Maximum number of iterations (default: 1000).
///
/// @return `SolverResult` on convergence, `SolverError` on failure.
template <concepts::Matrix ADerived, concepts::Vector BDerived,
          concepts::Vector XDerived>
auto bicgstab(const ADerived &A, const BDerived &b, const XDerived &x0,
              typename std::decay_t<ADerived>::value_type tol = 1e-10,
              index_type max_iter = 1000) {
  using AType = std::decay_t<ADerived>;
  using T = typename AType::value_type;
  constexpr index_type Dim = AType::extents_type::static_extent(0);
  using Vec = Vector<T, Dim>;
  using Result = SolverResult<T, Dim>;

  Vec x(x0);
  Vec r(b - A * x);

  T rnorm = r.norm();
  if (rnorm <= tol) {
    return std::expected<Result, SolverError>{
        Result{.x = std::move(x), .residual_norm = rnorm, .iterations = 0}};
  }

  // Shadow residual — kept constant throughout the iteration.
  Vec r_hat(r);

  Vec p(r);

  T rho = utils::detail::dot(r_hat, r);

  for (index_type iter = 0; iter < max_iter; ++iter) {
    // v = A * p
    Vec v(A * p);

    T r_hat_v = utils::detail::dot(r_hat, v);
    if (std::abs(r_hat_v) < std::numeric_limits<T>::min()) {
      return std::expected<Result, SolverError>{std::unexpected(SolverError{
          .kind = SolverError::Kind::breakdown,
          .message = "BiCGSTAB breakdown: r_hat^T A p is near zero",
      })};
    }

    T alpha = rho / r_hat_v;

    // s = r - alpha * v
    Vec s(r - alpha * v);

    T s_norm = s.norm();
    if (s_norm <= tol) {
      // The half-step already converged.
      x += alpha * p;
      return std::expected<Result, SolverError>{Result{
          .x = std::move(x), .residual_norm = s_norm, .iterations = iter + 1}};
    }

    // t = A * s
    Vec t(A * s);

    T t_dot_t = utils::detail::dot(t, t);
    if (t_dot_t < std::numeric_limits<T>::min()) {
      return std::expected<Result, SolverError>{std::unexpected(SolverError{
          .kind = SolverError::Kind::breakdown,
          .message = "BiCGSTAB breakdown: ||A*s||^2 is near zero",
      })};
    }

    T omega = utils::detail::dot(t, s) / t_dot_t;

    // x_{k+1} = x_k + alpha * p + omega * s
    x += alpha * p + omega * s;

    // r_{k+1} = s - omega * t
    r = Vec(s - omega * t);

    rnorm = r.norm();
    if (rnorm <= tol) {
      return std::expected<Result, SolverError>{Result{
          .x = std::move(x), .residual_norm = rnorm, .iterations = iter + 1}};
    }

    if (rnorm > T{1e30}) {
      return std::expected<Result, SolverError>{std::unexpected(SolverError{
          .kind = SolverError::Kind::diverged,
          .message = "BiCGSTAB diverged: residual norm exceeded 1e30",
      })};
    }

    T rho_new = utils::detail::dot(r_hat, r);
    if (std::abs(rho_new) < std::numeric_limits<T>::min()) {
      return std::expected<Result, SolverError>{std::unexpected(SolverError{
          .kind = SolverError::Kind::breakdown,
          .message = "BiCGSTAB breakdown: r_hat^T r is near zero "
                     "(shadow residual orthogonal to residual)",
      })};
    }

    if (std::abs(omega) < std::numeric_limits<T>::min()) {
      return std::expected<Result, SolverError>{std::unexpected(SolverError{
          .kind = SolverError::Kind::breakdown,
          .message = "BiCGSTAB breakdown: omega is near zero",
      })};
    }

    T beta = (rho_new / rho) * (alpha / omega);
    rho = rho_new;

    // p_{k+1} = r_{k+1} + beta * (p_k - omega * v)
    // Aliasing: Vec(...) evaluates fully before assignment.
    p = Vec(r + beta * (p - omega * v));
  }

  return std::expected<Result, SolverError>{std::unexpected(SolverError{
      .kind = SolverError::Kind::diverged,
      .message = "BiCGSTAB did not converge within max_iter iterations",
  })};
}

/// @brief BiCGSTAB starting from the zero vector.
///
/// Convenience overload -- see the other overload for full documentation.
template <concepts::Matrix ADerived, concepts::Vector BDerived>
auto bicgstab(const ADerived &A, const BDerived &b,
              typename std::decay_t<ADerived>::value_type tol = 1e-10,
              index_type max_iter = 1000) {
  using AType = std::decay_t<ADerived>;
  using T = typename AType::value_type;
  constexpr index_type Dim = AType::extents_type::static_extent(0);

  const index_type n = A.extent(0);
  Vector<T, Dim> x0(n);
  x0 = expression::nullary::Constant(T{0}, x0.extents());

  return bicgstab(A, b, x0, tol, max_iter);
}

} // namespace zipper::utils::solver

#endif
