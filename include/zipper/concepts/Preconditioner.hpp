/// @file Preconditioner.hpp
/// @brief Concept for preconditioners used with iterative solvers.
///
/// A `Preconditioner` is any type that can apply an approximate inverse of a
/// matrix to a vector.  It exposes:
///   - `value_type` -- the scalar type.
///   - `.apply(v)` -- returns a vector representing M^{-1} * v.
///
/// Preconditioners are used with the Preconditioned Conjugate Gradient (PCG)
/// solver to accelerate convergence by clustering eigenvalues.
///
/// @see zipper::utils::solver::JacobiPreconditioner
/// @see zipper::utils::solver::SSORPreconditioner
/// @see zipper::utils::solver::preconditioned_conjugate_gradient

#pragma once

#include <type_traits>

#include <zipper/Vector.hpp>

namespace zipper::concepts {

namespace preconditioner_detail {

/// Helper to check if a type has an `.apply(v)` method accepting a dynamic
/// vector of its value_type.
template <typename P, typename = void>
struct HasApplyMethod : std::false_type {};

template <typename P>
struct HasApplyMethod<
    P, std::void_t<decltype(std::declval<const P &>().apply(
           std::declval<
               const zipper::Vector<typename P::value_type,
                                    zipper::dynamic_extent> &>()))>>
    : std::true_type {};

} // namespace preconditioner_detail

/// A preconditioner can apply an approximate inverse M^{-1} to a vector.
///
/// Requirements:
///   - `P::value_type` exists.
///   - `p.apply(v)` is a valid expression for a `const P&` and a dynamic-extent
///     vector of the preconditioner's scalar type.
template <typename P>
concept Preconditioner = requires {
    typename P::value_type;
} && preconditioner_detail::HasApplyMethod<P>::value;

} // namespace zipper::concepts
