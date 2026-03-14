/// @file power_method.hpp
/// @brief Power iteration for computing the dominant eigenvalue/eigenvector
///        of a square matrix.
///
/// The power method repeatedly multiplies a vector by M and normalises,
/// converging to the eigenvector corresponding to the eigenvalue of largest
/// magnitude (the "dominant" eigenvalue).  The eigenvalue is estimated via the
/// Rayleigh quotient: lambda = v^T M v.
///
/// This is the simplest iterative eigensolver.  It is useful when only the
/// single largest eigenvalue is needed and the eigenvalue gap is large.  For
/// computing multiple eigenvalues or when convergence is slow, prefer the
/// Lanczos (symmetric) or Arnoldi (general) iterations -- see lanczos.hpp and
/// arnoldi.hpp.
///
/// Two public overloads are provided:
///
///   1. power_method(M, initial_guess, tol, max_iter)
///      -- user-supplied starting vector.
///
///   2. power_method(M, tol, max_iter)
///      -- convenience overload that generates a uniform random starting
///         vector.  A random vector is unlikely to be orthogonal to the
///         dominant eigenvector, so convergence is generally reliable.
///
/// Both return std::expected<PowerMethodResult, PowerMethodError> so that
/// convergence failure is reported as a value rather than an exception.

#if !defined(ZIPPER_UTILS_KRYLOV_POWER_METHOD_HPP)
#define ZIPPER_UTILS_KRYLOV_POWER_METHOD_HPP

#include <cmath>
#include <expected>
#include <limits>
#include <string>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Random.hpp>

namespace zipper::utils::krylov {

/// Result of a successful power iteration.
/// Contains the dominant eigenvalue and its corresponding eigenvector.
///
/// Template parameters:
///   T  -- scalar type (e.g. float, double).
///   N  -- compile-time vector dimension (or dynamic_extent if only known at
///         runtime).
template <typename T, index_type N> struct PowerMethodResult {
  T eigenvalue;
  Vector<T, N> eigenvector;
};

/// Error type returned when power iteration fails to converge.
struct PowerMethodError {
  std::string message;
};

namespace detail {

/// Core power iteration loop, shared by both public overloads.
///
/// This internal helper separates the iteration logic from input validation
/// and initial-vector generation so that the public overloads can share the
/// algorithm without duplication.
///
/// @param M         Square matrix (n x n).
/// @param v         Initial vector of size n (non-zero; normalised internally).
/// @param tol       Convergence tolerance on the eigenvector change.
/// @param max_iter  Maximum number of iterations before declaring failure.
template <concepts::Matrix MatType, index_type Dim>
auto power_method_impl(const MatType &M,
                       Vector<typename MatType::value_type, Dim> v,
                       typename MatType::value_type tol, index_type max_iter) {
  using T = typename MatType::value_type;
  using VecType = Vector<T, Dim>;
  using Result = PowerMethodResult<T, Dim>;
  using Expected = std::expected<Result, PowerMethodError>;

  // Normalise initial vector; if it is the zero vector, fall back to e_0.
  {
    T nrm = v.norm();
    if (nrm == T{0}) {
      v(0) = T{1};
      nrm = T{1};
    }
    v = VecType(v / nrm);
  }

  T eigenvalue = T{0};

  for (index_type iter = 0; iter < max_iter; ++iter) {
    // Expand: w = M * v.
    VecType w(M * v);

    T nrm = w.norm();

    if (nrm == T{0}) {
      // M annihilates v -- eigenvalue is 0.
      return Expected(Result{
          .eigenvalue = T{0},
          .eigenvector = std::move(v),
      });
    }

    // Normalise to get the new eigenvector estimate.
    VecType v_new(w / nrm);

    // Rayleigh quotient for the eigenvalue: lambda = v^T M v.
    T lambda = v_new.dot(VecType(M * v_new));

    // Check convergence: ||v_new - v|| < tol.
    // Account for sign flip: v_new and -v_new are both valid eigenvectors,
    // so we take the minimum of both differences to avoid a false
    // non-convergence when the sign oscillates.
    VecType diff(v_new - v);
    VecType diff_flip(v_new + v);
    T delta = std::min(diff.norm(), diff_flip.norm());

    eigenvalue = lambda;
    v = std::move(v_new);

    if (delta < tol) {
      return Expected(Result{
          .eigenvalue = eigenvalue,
          .eigenvector = std::move(v),
      });
    }
  }

  return Expected(std::unexpected(
      PowerMethodError{"power_method did not converge within " +
                       std::to_string(max_iter) + " iterations"}));
}

} // namespace detail

/// Power method with a user-provided initial guess vector.
///
/// @param M              A square matrix.
/// @param initial_guess  Starting vector for iteration.
/// @param tol            Convergence tolerance (default 1e-10).
/// @param max_iter       Maximum number of iterations (default 1000).
template <concepts::Matrix MatType, concepts::Vector VecType>
auto power_method(const MatType &M, const VecType &initial_guess,
                  typename MatType::value_type tol = 1e-10,
                  index_type max_iter = 1000) {
  using T = typename MatType::value_type;
  using MType = std::decay_t<MatType>;

  constexpr index_type StaticRows = MType::extents_type::static_extent(0);

  // Compile-time square-matrix check when both extents are static;
  // runtime assert otherwise.
  if constexpr (MType::extents_traits::is_static) {
    constexpr index_type StaticCols = MType::extents_type::static_extent(1);
    static_assert(StaticRows == StaticCols,
                  "power_method requires a square matrix");
  }

  constexpr index_type Dim = StaticRows;
  using ResultVec = Vector<T, Dim>;
  using Expected = std::expected<PowerMethodResult<T, Dim>, PowerMethodError>;

  const index_type n = M.extent(0);
  if (n != M.extent(1)) {
    return Expected(std::unexpected(
        PowerMethodError{"power_method requires a square matrix"}));
  }
  if (n == 0) {
    return Expected(std::unexpected(
        PowerMethodError{"power_method requires a non-empty matrix"}));
  }

  ResultVec v(initial_guess);
  return detail::power_method_impl(M, std::move(v), tol, max_iter);
}

/// Power method with a uniform random initial vector (convenience).
///
/// Generates a random starting vector and delegates to the core iteration.
/// A random vector is unlikely to be orthogonal to the dominant eigenvector,
/// so convergence is generally reliable.
///
/// @param M          A square matrix.
/// @param tol        Convergence tolerance (default 1e-10).
/// @param max_iter   Maximum number of iterations (default 1000).
template <concepts::Matrix MatType>
auto power_method(const MatType &M, typename MatType::value_type tol = 1e-10,
                  index_type max_iter = 1000) {
  using T = typename MatType::value_type;
  using MType = std::decay_t<MatType>;

  constexpr index_type StaticRows = MType::extents_type::static_extent(0);

  if constexpr (MType::extents_traits::is_static) {
    constexpr index_type StaticCols = MType::extents_type::static_extent(1);
    static_assert(StaticRows == StaticCols,
                  "power_method requires a square matrix");
  }

  constexpr index_type Dim = StaticRows;
  using ResultVec = Vector<T, Dim>;
  using Expected = std::expected<PowerMethodResult<T, Dim>, PowerMethodError>;

  const index_type n = M.extent(0);
  if (n != M.extent(1)) {
    return Expected(std::unexpected(
        PowerMethodError{"power_method requires a square matrix"}));
  }
  if (n == 0) {
    return Expected(std::unexpected(
        PowerMethodError{"power_method requires a non-empty matrix"}));
  }

  // Random starting vector in [-1, 1]; unlikely to be orthogonal to the
  // dominant eigenvector.
  ResultVec v(expression::nullary::uniform_random<T>(zipper::extents<Dim>(n),
                                                     T{-1}, T{1}));
  return detail::power_method_impl(M, std::move(v), tol, max_iter);
}

} // namespace zipper::utils::krylov

#endif
