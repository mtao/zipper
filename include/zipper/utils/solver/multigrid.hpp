/// @file multigrid.hpp
/// @brief Geometric multigrid V-cycle solver for linear systems Ax = b.
/// @ingroup solvers
///
/// Multigrid methods accelerate convergence of iterative solvers by
/// operating on a hierarchy of grids at different resolutions.  High-frequency
/// error components are eliminated by smoothing on the fine grid, while
/// low-frequency components are corrected by solving a coarser problem.
///
/// This implementation provides:
///   - `MultigridLevel<T>` -- a single level of the hierarchy, storing the
///     system matrix A, restriction operator R, and prolongation operator P.
///   - `vcycle(...)` -- a single V-cycle iteration.
///   - `multigrid(...)` -- the outer iteration that repeatedly applies V-cycles
///     until convergence.
///   - `build_1d_hierarchy(...)` -- a helper to construct a geometric multigrid
///     hierarchy for a 1D finite-difference Laplacian (or any tridiagonal SPD
///     system on a 2^k - 1 grid).
///
/// The V-cycle proceeds as:
///   1. Pre-smooth (damped Jacobi) on the fine grid.
///   2. Compute residual r = b - A*x, restrict to coarse grid: r_c = R * r.
///   3. Recursively solve A_c * e_c = r_c on the coarse grid (or direct-solve
///      at the coarsest level via PLU).
///   4. Prolongate correction: x += P * e_c.
///   5. Post-smooth (damped Jacobi) on the fine grid.
///
/// At the coarsest level (smallest grid), the system is solved exactly using
/// PLU decomposition.
///
/// The smoother is damped (weighted) Jacobi with a configurable relaxation
/// parameter omega (default 2/3, which is optimal for the 1D Laplacian).
///
/// @see zipper::utils::solver::SolverResult
/// @see zipper::utils::solver::SolverError
/// @see zipper::utils::decomposition::plu

#if !defined(ZIPPER_UTILS_SOLVER_MULTIGRID_HPP)
#define ZIPPER_UTILS_SOLVER_MULTIGRID_HPP

#include <cmath>
#include <vector>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/utils/decomposition/lu.hpp>
#include <zipper/utils/detail/dot.hpp>

#include "result.hpp"

namespace zipper::utils::solver {

// ─────────────────────────────────────────────────────────────────────────────
// Hierarchy types
// ─────────────────────────────────────────────────────────────────────────────

/// @brief A single level of the multigrid hierarchy.
///
/// Stores the system matrix A at this level, along with the restriction
/// operator R (fine-to-coarse) and prolongation operator P (coarse-to-fine).
/// At the coarsest level, R and P are unused (they would map to a
/// non-existent even-coarser level).
///
/// @tparam T  Scalar type.
template <typename T> struct MultigridLevel {
  using value_type = T;

  /// System matrix at this level.
  MatrixXX<T> A;
  /// Restriction operator (fine -> coarse).
  MatrixXX<T> R;
  /// Prolongation operator (coarse -> fine).
  MatrixXX<T> P;
};

// ─────────────────────────────────────────────────────────────────────────────
// Smoother (damped Jacobi)
// ─────────────────────────────────────────────────────────────────────────────

namespace multigrid_detail {

/// @brief Perform `num_sweeps` damped Jacobi smoothing iterations.
///
/// x <- x + omega * D^{-1} * (b - A*x)
///
/// @param A          System matrix.
/// @param b          Right-hand side.
/// @param x          Current solution (modified in place).
/// @param omega      Damping parameter (typically 2/3).
/// @param num_sweeps Number of smoothing sweeps.
template <typename T>
void damped_jacobi_smooth(const MatrixXX<T> &A, const VectorX<T> &b,
                          VectorX<T> &x, T omega, index_type num_sweeps) {
  const index_type n = A.rows();
  for (index_type sweep = 0; sweep < num_sweeps; ++sweep) {
    VectorX<T> x_new(n);
    for (index_type i = 0; i < n; ++i) {
      T sigma = T{0};
      for (index_type j = 0; j < n; ++j) {
        sigma += A(i, j) * x(j);
      }
      // x_new_i = x_i + omega * (b_i - (A*x)_i) / A_ii
      x_new(i) = x(i) + omega * (b(i) - sigma) / A(i, i);
    }
    x = std::move(x_new);
  }
}

} // namespace multigrid_detail

// ─────────────────────────────────────────────────────────────────────────────
// V-cycle
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Perform a single V-cycle on the multigrid hierarchy.
///
/// @param hierarchy  Vector of MultigridLevel, ordered finest (index 0) to
///                   coarsest (last index).
/// @param level      Current level index (0 = finest).
/// @param b          Right-hand side at this level.
/// @param x          Current solution at this level (modified in place).
/// @param omega      Damping parameter for the smoother.
/// @param pre_sweeps Number of pre-smoothing sweeps.
/// @param post_sweeps Number of post-smoothing sweeps.
///
/// @return true on success, false if the coarsest-level direct solve fails.
template <typename T>
auto vcycle(const std::vector<MultigridLevel<T>> &hierarchy,
            index_type level, const VectorX<T> &b, VectorX<T> &x, T omega,
            index_type pre_sweeps, index_type post_sweeps) -> bool {
  const index_type num_levels = static_cast<index_type>(hierarchy.size());

  // Base case: coarsest level -- direct solve via PLU.
  if (level == num_levels - 1) {
    auto result = decomposition::plu_solve(hierarchy[level].A, b);
    if (!result) {
      return false;
    }
    x = std::move(*result);
    return true;
  }

  const auto &lvl = hierarchy[level];

  // 1. Pre-smooth.
  multigrid_detail::damped_jacobi_smooth(lvl.A, b, x, omega, pre_sweeps);

  // 2. Compute residual and restrict.
  VectorX<T> r(b - lvl.A * x);
  VectorX<T> r_coarse(lvl.R * r);

  // 3. Recurse on coarse grid (zero initial guess for the correction).
  const index_type n_coarse = hierarchy[level + 1].A.rows();
  VectorX<T> e_coarse(n_coarse);
  e_coarse = expression::nullary::Constant(T{0}, e_coarse.extents());

  if (!vcycle(hierarchy, level + 1, r_coarse, e_coarse, omega, pre_sweeps,
              post_sweeps)) {
    return false;
  }

  // 4. Prolongate correction and apply.
  VectorX<T> correction(lvl.P * e_coarse);
  x += correction;

  // 5. Post-smooth.
  multigrid_detail::damped_jacobi_smooth(lvl.A, b, x, omega, post_sweeps);

  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Outer iteration
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Multigrid solver: repeatedly applies V-cycles until convergence.
///
/// @param hierarchy   Vector of MultigridLevel, finest to coarsest.
/// @param b           Right-hand side vector (at the finest level).
/// @param x0          Initial guess.
/// @param tol         Convergence tolerance on ||r||_2.
/// @param max_iter    Maximum number of V-cycle iterations.
/// @param omega       Damping parameter for the smoother (default: 2/3).
/// @param pre_sweeps  Number of pre-smoothing sweeps per V-cycle (default: 2).
/// @param post_sweeps Number of post-smoothing sweeps per V-cycle (default: 2).
///
/// @return `SolverResult` on convergence, `SolverError` on failure.
template <typename T>
auto multigrid(const std::vector<MultigridLevel<T>> &hierarchy,
               const VectorX<T> &b, const VectorX<T> &x0, T tol = T{1e-10},
               index_type max_iter = 100, T omega = T{2} / T{3},
               index_type pre_sweeps = 2, index_type post_sweeps = 2) {
  using Vec = VectorX<T>;
  using Result = SolverResult<T, dynamic_extent>;

  Vec x(x0);

  for (index_type iter = 0; iter < max_iter; ++iter) {
    // Check convergence before the cycle.
    Vec r(b - hierarchy[0].A * x);
    T rnorm = std::sqrt(utils::detail::dot(r, r));

    if (rnorm <= tol) {
      return std::expected<Result, SolverError>{
          Result{.x = std::move(x), .residual_norm = rnorm, .iterations = iter}};
    }

    // Perform one V-cycle.
    if (!vcycle(hierarchy, index_type{0}, b, x, omega, pre_sweeps,
                post_sweeps)) {
      return std::expected<Result, SolverError>{std::unexpected(SolverError{
          .kind = SolverError::Kind::breakdown,
          .message = "Multigrid: coarsest-level direct solve failed",
      })};
    }
  }

  // Check one final time after the last cycle.
  Vec r(b - hierarchy[0].A * x);
  T rnorm = std::sqrt(utils::detail::dot(r, r));
  if (rnorm <= tol) {
    return std::expected<Result, SolverError>{
        Result{.x = std::move(x), .residual_norm = rnorm, .iterations = max_iter}};
  }

  return std::expected<Result, SolverError>{std::unexpected(SolverError{
      .kind = SolverError::Kind::diverged,
      .message = "Multigrid did not converge within max_iter V-cycles",
  })};
}

/// @brief Multigrid solver with zero initial guess.
///
/// Convenience overload -- see the other overload for full documentation.
template <typename T>
auto multigrid(const std::vector<MultigridLevel<T>> &hierarchy,
               const VectorX<T> &b, T tol = T{1e-10},
               index_type max_iter = 100, T omega = T{2} / T{3},
               index_type pre_sweeps = 2, index_type post_sweeps = 2) {
  const index_type n = hierarchy[0].A.rows();
  VectorX<T> x0(n);
  x0 = expression::nullary::Constant(T{0}, x0.extents());

  return multigrid(hierarchy, b, x0, tol, max_iter, omega, pre_sweeps,
                   post_sweeps);
}

// ─────────────────────────────────────────────────────────────────────────────
// 1D hierarchy builder
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Build a geometric multigrid hierarchy for a 1D problem.
///
/// Constructs a hierarchy of levels for a 1D finite-difference problem on
/// n = 2^k - 1 interior points.  At each level:
///   - Restriction R is full-weighting: R_{i,2i} = 1/4, R_{i,2i+1} = 1/2,
///     R_{i,2i+2} = 1/4 (maps n_fine -> n_coarse = (n_fine-1)/2).
///   - Prolongation P = 2 * R^T (linear interpolation).
///   - Coarse system matrix A_coarse = R * A_fine * P (Galerkin condition).
///
/// The finest-level matrix A_fine is provided by the caller (typically the
/// 1D Laplacian: tridiagonal with -1, 2, -1 scaled by (n+1)^2).
///
/// @param A_fine  The system matrix at the finest level (n x n, n = 2^k - 1).
/// @param num_levels Number of levels in the hierarchy (including finest).
///                   Must satisfy: n >= 2^(num_levels-1) - 1, i.e. the
///                   coarsest level has at least 1 interior point.
///
/// @return Vector of MultigridLevel, finest (index 0) to coarsest.
template <typename T>
auto build_1d_hierarchy(const MatrixXX<T> &A_fine, index_type num_levels)
    -> std::vector<MultigridLevel<T>> {
  std::vector<MultigridLevel<T>> hierarchy;
  hierarchy.reserve(num_levels);

  MatrixXX<T> A_current(A_fine);

  for (index_type lvl = 0; lvl < num_levels; ++lvl) {
    const index_type n_fine = A_current.rows();

    if (lvl == num_levels - 1) {
      // Coarsest level: no R/P needed (they won't be used).
      // Store zero-sized R/P matrices as placeholders.
      MatrixXX<T> R_dummy(0, 0);
      MatrixXX<T> P_dummy(0, 0);
      hierarchy.push_back(MultigridLevel<T>{
          .A = std::move(A_current),
          .R = std::move(R_dummy),
          .P = std::move(P_dummy),
      });
      break;
    }

    // Coarse grid size: n_coarse = (n_fine - 1) / 2.
    const index_type n_coarse = (n_fine - 1) / 2;

    // Build full-weighting restriction operator R (n_coarse x n_fine).
    MatrixXX<T> R(n_coarse, n_fine);
    R = expression::nullary::Constant(T{0}, R.extents());
    for (index_type i = 0; i < n_coarse; ++i) {
      const index_type j = 2 * i; // fine-grid index
      R(i, j) = T{0.25};
      R(i, j + 1) = T{0.5};
      R(i, j + 2) = T{0.25};
    }

    // Prolongation P = 2 * R^T (n_fine x n_coarse).
    // Build explicitly for clarity.
    MatrixXX<T> P(n_fine, n_coarse);
    P = expression::nullary::Constant(T{0}, P.extents());
    for (index_type i = 0; i < n_coarse; ++i) {
      const index_type j = 2 * i;
      P(j, i) = T{0.5};
      P(j + 1, i) = T{1.0};
      P(j + 2, i) = T{0.5};
    }

    // Galerkin coarsening: A_coarse = R * A_fine * P.
    MatrixXX<T> RAP(R * MatrixXX<T>(A_current * P));

    hierarchy.push_back(MultigridLevel<T>{
        .A = std::move(A_current),
        .R = std::move(R),
        .P = std::move(P),
    });

    A_current = std::move(RAP);
  }

  return hierarchy;
}

} // namespace zipper::utils::solver

#endif
