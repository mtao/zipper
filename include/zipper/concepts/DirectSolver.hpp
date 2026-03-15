/// @file DirectSolver.hpp
/// @brief Concept for direct (decomposition-based) linear solvers.
///
/// A `DirectSolver` is any type that stores a pre-computed decomposition of a
/// matrix and exposes a `.solve(b)` method to solve the system Ax = b for a
/// given right-hand side vector b.
///
/// This concept is satisfied by the result types of the library's
/// decomposition functions:
///   - `LLTResult`         (Cholesky)
///   - `LDLTResult`        (LDLT)
///   - `QRReducedResult`   (reduced QR)
///   - `QRFullResult`      (full QR)
///
/// The concept is structural (constraint-based): it checks for a `value_type`
/// member type and the existence of a `.solve(b)` method that accepts a
/// dynamic-extent vector (which is always dimension-compatible at runtime).
///
/// @see zipper::utils::decomposition::LLTResult
/// @see zipper::utils::decomposition::LDLTResult
/// @see zipper::utils::decomposition::QRReducedResult
/// @see zipper::utils::decomposition::QRFullResult

#pragma once

#include <type_traits>

#include <zipper/Vector.hpp>

namespace zipper::concepts {

namespace direct_solver_detail {

/// Helper to check if a type has a `.solve(b)` method accepting a dynamic
/// vector of its value_type.
template <typename S, typename = void>
struct HasSolveMethod : std::false_type {};

template <typename S>
struct HasSolveMethod<
    S, std::void_t<decltype(std::declval<const S &>().solve(
           std::declval<
               const zipper::Vector<typename S::value_type,
                                    zipper::dynamic_extent> &>()))>>
    : std::true_type {};

} // namespace direct_solver_detail

/// A direct solver stores a decomposition and can solve Ax = b.
///
/// Requirements:
///   - `S::value_type` exists.
///   - `s.solve(b)` is a valid expression for a `const S&` and a dynamic-extent
///     vector of the solver's scalar type.
template <typename S>
concept DirectSolver = requires {
    typename S::value_type;
} && direct_solver_detail::HasSolveMethod<S>::value;

} // namespace zipper::concepts
