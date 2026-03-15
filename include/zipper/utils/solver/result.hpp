/// @file result.hpp
/// @brief Common result and error types for iterative linear solvers.
/// @ingroup solvers
///
/// All iterative solvers in `zipper::utils::solver` return
/// `std::expected<SolverResult<T, Dim>, SolverError>`.  On success the result
/// contains the approximate solution vector, the final residual norm, and the
/// number of iterations performed.  On failure a `SolverError` describes what
/// went wrong (e.g. the iteration diverged or a required structural property
/// of the matrix was violated).
///
/// The `SolverResult` template is parameterised on the scalar type T and the
/// compile-time vector dimension Dim (which may be `dynamic_extent` when the
/// size is only known at runtime).  This mirrors the zipper convention used
/// throughout the krylov and orthogonalization modules.
///
/// @see zipper::utils::solver::conjugate_gradient — CG solver (SPD matrices).
/// @see zipper::utils::solver::gauss_seidel — Gauss-Seidel iteration.
/// @see zipper::utils::solver::gmres — GMRES solver (general matrices).
/// @see zipper::utils::solver::bicgstab — BiCGSTAB solver (general matrices).
/// @see zipper::utils::solver::triangular_solve — direct triangular solver
///      (returns `std::expected<Vector, SolverError>` rather than SolverResult).

#if !defined(ZIPPER_UTILS_SOLVER_RESULT_HPP)
#define ZIPPER_UTILS_SOLVER_RESULT_HPP

#include <expected>
#include <string>

#include <zipper/Vector.hpp>

namespace zipper::utils::solver {

/// Describes what went wrong when an iterative solver fails.
///
/// Currently two failure modes are distinguished:
///   - `diverged`:  the residual norm grew beyond a reasonable bound, or the
///                  iteration did not converge within the allowed number of
///                  steps.
///   - `breakdown`: a numerical breakdown occurred (e.g. a zero pivot in
///                  Gauss-Seidel, or a zero denominator in BiCGSTAB).
///
/// A human-readable message is always provided for diagnostic purposes.
struct SolverError {
  enum class Kind {
    diverged,
    breakdown,
  };

  Kind kind;
  std::string message;
};

/// The successful result of an iterative linear solver.
///
/// @tparam T    Scalar type (e.g. `float`, `double`).
/// @tparam Dim  Compile-time dimension of the solution vector, or
///              `dynamic_extent` for runtime-sized vectors.
///
/// Fields:
///   - `x`:             the approximate solution vector.
///   - `residual_norm`: the 2-norm of the residual  b - A*x  at termination.
///   - `iterations`:    the number of iterations actually performed (may be
///                      less than `max_iter` if the solver converged early).
template <typename T, index_type Dim = dynamic_extent> struct SolverResult {
  Vector<T, Dim> x;
  T residual_norm;
  index_type iterations;
};

} // namespace zipper::utils::solver

#endif
