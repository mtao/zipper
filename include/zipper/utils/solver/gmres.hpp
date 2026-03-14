/// @file gmres.hpp
/// @brief Generalized Minimum Residual (GMRES) method for general
///        non-symmetric linear systems Ax = b.
///
/// GMRES is the standard Krylov solver for general (non-symmetric, possibly
/// non-square in principle, though we require square here) linear systems.
/// It builds an orthonormal basis for the Krylov subspace using the Arnoldi
/// process and solves a least-squares problem in that subspace to minimise
/// the residual norm at each step.
///
/// The algorithm (full GMRES without restart):
///   1. Compute r_0 = b - A * x_0, beta = ||r_0||, q_1 = r_0 / beta.
///   2. For j = 1, 2, ..., m (up to max_iter):
///      a. Arnoldi step: w = A * q_j, orthogonalise against q_1..q_j,
///         compute h_{i,j} for i = 1..j and h_{j+1,j} = ||w||.
///      b. Apply previous Givens rotations to the new column of H.
///      c. Compute new Givens rotation to zero out h_{j+1,j}.
///      d. Update the residual norm estimate.
///      e. If converged, solve the upper triangular system and return.
///   3. If not converged after max_iter steps, return the best solution.
///
/// This implementation uses `std::vector` for the Krylov basis vectors and
/// the Hessenberg matrix columns, since the number of iterations (and hence
/// the basis size) is not known at compile time.
///
/// Storage cost: O(n * m) for the basis vectors, O(m^2) for the Hessenberg
/// matrix, where m is the number of iterations.  For very large m this can
/// be expensive; in production one would use restarted GMRES(k), but for
/// this library the unrestarted variant is clearer and sufficient.
///
/// Two overloads:
///   - `gmres(A, b, x0, tol, max_iter)` -- with initial guess.
///   - `gmres(A, b, tol, max_iter)` -- zero initial guess.

#if !defined(ZIPPER_UTILS_SOLVER_GMRES_HPP)
#define ZIPPER_UTILS_SOLVER_GMRES_HPP

#include <cmath>
#include <limits>
#include <vector>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/utils/detail/dot.hpp>

#include "result.hpp"

namespace zipper::utils::solver {

/// @brief GMRES with an explicit initial guess.
///
/// @param A        Square coefficient matrix (general, non-symmetric OK).
/// @param b        Right-hand side vector.
/// @param x0       Initial guess for the solution.
/// @param tol      Convergence tolerance on ||b - A*x||_2 (default: 1e-10).
/// @param max_iter Maximum number of Arnoldi steps (default: 1000).
///
/// @return `SolverResult` on convergence, `SolverError` otherwise.
template <concepts::Matrix ADerived, concepts::Vector BDerived,
          concepts::Vector XDerived>
auto gmres(const ADerived &A, const BDerived &b, const XDerived &x0,
           typename std::decay_t<ADerived>::value_type tol = 1e-10,
           index_type max_iter = 1000) {
  using AType = std::decay_t<ADerived>;
  using T = typename AType::value_type;
  constexpr index_type Dim = AType::extents_type::static_extent(0);
  using Vec = Vector<T, Dim>;
  using Result = SolverResult<T, Dim>;

  Vec x(x0);
  Vec r0(b - A * x);
  T beta = r0.norm();

  if (beta <= tol) {
    return std::expected<Result, SolverError>{
        Result{.x = std::move(x), .residual_norm = beta, .iterations = 0}};
  }

  // Krylov basis vectors: Q[0] = r0 / beta, Q[1], Q[2], ...
  std::vector<Vec> Q;
  Q.reserve(max_iter + 1);
  Q.emplace_back(r0 / beta);

  // Upper Hessenberg matrix H stored column-by-column.
  // H_col[j] has j+2 entries: H(0,j), H(1,j), ..., H(j+1,j).
  std::vector<std::vector<T>> H_col;
  H_col.reserve(max_iter);

  // Givens rotation parameters: cs[i] and sn[i] for the i-th rotation.
  std::vector<T> cs, sn;
  cs.reserve(max_iter);
  sn.reserve(max_iter);

  // Right-hand side of the least-squares problem: g = [beta, 0, 0, ...]^T.
  // We maintain it incrementally as Givens rotations are applied.
  std::vector<T> g;
  g.reserve(max_iter + 1);
  g.push_back(beta);

  for (index_type j = 0; j < max_iter; ++j) {
    // --- Arnoldi step: expand the Krylov subspace ---
    Vec w(A * Q[j]);

    std::vector<T> h(j + 2, T{0});

    // Orthogonalise w against Q[0..j] (classical Gram-Schmidt).
    for (index_type i = 0; i <= j; ++i) {
      h[i] = utils::detail::dot(Q[i], w);
      w -= h[i] * Q[i];
    }
    h[j + 1] = w.norm();

    // --- Apply previous Givens rotations to the new column of H ---
    for (index_type i = 0; i < j; ++i) {
      T temp = cs[i] * h[i] + sn[i] * h[i + 1];
      h[i + 1] = -sn[i] * h[i] + cs[i] * h[i + 1];
      h[i] = temp;
    }

    // --- Compute new Givens rotation to eliminate h[j+1] ---
    T h_jj = h[j];
    T h_jp1j = h[j + 1];
    T denom = std::hypot(h_jj, h_jp1j);

    if (denom < std::numeric_limits<T>::min()) {
      return std::expected<Result, SolverError>{std::unexpected(SolverError{
          .kind = SolverError::Kind::breakdown,
          .message = "GMRES breakdown: Arnoldi produced a zero vector",
      })};
    }

    T c = h_jj / denom;
    T s = h_jp1j / denom;
    cs.push_back(c);
    sn.push_back(s);

    // Apply the new rotation to h and g.
    h[j] = denom;
    h[j + 1] = T{0};

    g.push_back(-s * g[j]);
    g[j] = c * g[j];

    H_col.push_back(std::move(h));

    // The residual norm estimate is |g[j+1]|.
    T res_est = std::abs(g[j + 1]);

    // Add the new basis vector (only if we might need another iteration).
    if (h_jp1j > std::numeric_limits<T>::epsilon()) {
      Q.emplace_back(w / h_jp1j);
    } else if (res_est > tol) {
      // Arnoldi breakdown but not yet converged -- push a dummy to keep
      // indices consistent, but we will not use it.
      Q.emplace_back(Q[0]); // placeholder
    }

    if (res_est <= tol || j == max_iter - 1) {
      // --- Back-substitution to solve H * y = g ---
      index_type m = j + 1; // number of Arnoldi steps completed
      std::vector<T> y(m, T{0});
      for (index_type k = m; k > 0; --k) {
        index_type ki = k - 1;
        y[ki] = g[ki];
        for (index_type l = ki + 1; l < m; ++l) {
          y[ki] -= H_col[l][ki] * y[l];
        }
        y[ki] /= H_col[ki][ki];
      }

      // x = x0 + Q * y
      for (index_type k = 0; k < m; ++k) {
        x += y[k] * Q[k];
      }

      // Compute actual residual for the result.
      Vec r_final(b - A * x);
      T rnorm = r_final.norm();

      if (res_est <= tol) {
        return std::expected<Result, SolverError>{Result{
            .x = std::move(x), .residual_norm = rnorm, .iterations = m}};
      } else {
        return std::expected<Result, SolverError>{
            std::unexpected(SolverError{
                .kind = SolverError::Kind::diverged,
                .message =
                    "GMRES did not converge within max_iter iterations",
            })};
      }
    }
  }

  // Should not reach here, but just in case.
  Vec r_final(b - A * x);
  return std::expected<Result, SolverError>{std::unexpected(SolverError{
      .kind = SolverError::Kind::diverged,
      .message = "GMRES did not converge",
  })};
}

/// @brief GMRES starting from the zero vector.
///
/// Convenience overload -- see the other overload for full documentation.
template <concepts::Matrix ADerived, concepts::Vector BDerived>
auto gmres(const ADerived &A, const BDerived &b,
           typename std::decay_t<ADerived>::value_type tol = 1e-10,
           index_type max_iter = 1000) {
  using AType = std::decay_t<ADerived>;
  using T = typename AType::value_type;
  constexpr index_type Dim = AType::extents_type::static_extent(0);

  const index_type n = A.extent(0);
  Vector<T, Dim> x0(n);
  x0 = expression::nullary::Constant(T{0}, x0.extents());

  return gmres(A, b, x0, tol, max_iter);
}

} // namespace zipper::utils::solver

#endif
