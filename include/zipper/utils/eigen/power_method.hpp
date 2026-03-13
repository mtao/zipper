
#if !defined(ZIPPER_UTILS_EIGEN_POWER_METHOD_HPP)
#define ZIPPER_UTILS_EIGEN_POWER_METHOD_HPP

#include <cmath>
#include <expected>
#include <stdexcept>
#include <string>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/concepts/Matrix.hpp>
#include <zipper/concepts/Vector.hpp>
#include <zipper/expression/nullary/Random.hpp>

namespace zipper::utils::eigen {

/// Result of a successful power iteration.
/// Contains the dominant eigenvalue and its corresponding eigenvector.
template <typename T, index_type N>
struct PowerMethodResult {
    T eigenvalue;
    Vector<T, N> eigenvector;
};

/// Error type returned when power iteration fails to converge.
struct PowerMethodError {
    std::string message;
};

namespace detail {

/// Core power iteration loop, shared by both overloads.
/// Expects v to be a non-zero vector of size n.
template <concepts::Matrix MatType, index_type Dim>
auto power_method_impl(const MatType& M,
                       Vector<typename MatType::value_type, Dim> v,
                       typename MatType::value_type tol,
                       index_type max_iter) {
    using T = typename MatType::value_type;
    using VecType = Vector<T, Dim>;
    using Result = PowerMethodResult<T, Dim>;
    using Expected = std::expected<Result, PowerMethodError>;

    // Normalize initial vector
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
        // w = M * v
        VecType w(M * v);

        T nrm = w.norm();

        if (nrm == T{0}) {
            // M annihilates v — eigenvalue is 0
            return Expected(Result{
                .eigenvalue = T{0},
                .eigenvector = std::move(v),
            });
        }

        // Normalize to get new eigenvector estimate
        VecType v_new(w / nrm);

        // Rayleigh quotient for eigenvalue: lambda = v^T * M * v
        T lambda = v_new.dot(VecType(M * v_new));

        // Check convergence: ||v_new - v|| < tol
        // Account for sign flip: v_new and -v_new are both valid
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

    return Expected(std::unexpected(PowerMethodError{
        "power_method did not converge within " +
        std::to_string(max_iter) + " iterations"}));
}

}  // namespace detail

/// Power method with a user-provided initial guess vector.
///
/// @param M              A square matrix
/// @param initial_guess  Starting vector for iteration
/// @param tol            Convergence tolerance (default 1e-10)
/// @param max_iter       Maximum number of iterations (default 1000)
template <concepts::Matrix MatType, concepts::Vector VecType>
auto power_method(const MatType& M,
                  const VecType& initial_guess,
                  typename MatType::value_type tol = 1e-10,
                  index_type max_iter = 1000) {
    using T = typename MatType::value_type;
    using extents_type = typename MatType::extents_type;
    constexpr index_type StaticRows = extents_type::static_extent(0);
    constexpr index_type StaticCols = extents_type::static_extent(1);

    if constexpr (StaticRows != std::dynamic_extent &&
                  StaticCols != std::dynamic_extent) {
        static_assert(StaticRows == StaticCols,
                      "power_method requires a square matrix");
    }

    constexpr index_type Dim = StaticRows;
    using ResultVec = Vector<T, Dim>;
    using Expected = std::expected<PowerMethodResult<T, Dim>, PowerMethodError>;

    const index_type n = M.extent(0);
    if (n != M.extent(1)) {
        return Expected(std::unexpected(PowerMethodError{
            "power_method requires a square matrix"}));
    }
    if (n == 0) {
        return Expected(std::unexpected(PowerMethodError{
            "power_method requires a non-empty matrix"}));
    }

    ResultVec v(initial_guess);
    return detail::power_method_impl(M, std::move(v), tol, max_iter);
}

/// Power method with a uniform random initial vector (recommended).
///
/// @param M          A square matrix
/// @param tol        Convergence tolerance (default 1e-10)
/// @param max_iter   Maximum number of iterations (default 1000)
template <concepts::Matrix MatType>
auto power_method(const MatType& M,
                  typename MatType::value_type tol = 1e-10,
                  index_type max_iter = 1000) {
    using T = typename MatType::value_type;
    using extents_type = typename MatType::extents_type;
    constexpr index_type StaticRows = extents_type::static_extent(0);
    constexpr index_type StaticCols = extents_type::static_extent(1);

    if constexpr (StaticRows != std::dynamic_extent &&
                  StaticCols != std::dynamic_extent) {
        static_assert(StaticRows == StaticCols,
                      "power_method requires a square matrix");
    }

    constexpr index_type Dim = StaticRows;
    using ResultVec = Vector<T, Dim>;
    using Expected = std::expected<PowerMethodResult<T, Dim>, PowerMethodError>;

    const index_type n = M.extent(0);
    if (n != M.extent(1)) {
        return Expected(std::unexpected(PowerMethodError{
            "power_method requires a square matrix"}));
    }
    if (n == 0) {
        return Expected(std::unexpected(PowerMethodError{
            "power_method requires a non-empty matrix"}));
    }

    ResultVec v(expression::nullary::uniform_random<T>(
        zipper::extents<Dim>(n), T{-1}, T{1}));
    return detail::power_method_impl(M, std::move(v), tol, max_iter);
}

}  // namespace zipper::utils::eigen
#endif
