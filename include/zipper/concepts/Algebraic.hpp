#if !defined(ZIPPER_CONCEPTS_ALGEBRAIC_HPP)
#define ZIPPER_CONCEPTS_ALGEBRAIC_HPP

/// @file Algebraic.hpp
/// @brief Algebraic structure concepts for scalar types.
///
/// Provides concepts that classify scalar types by their algebraic
/// properties, following the standard mathematical hierarchy:
///
///   ring  ⊂  field
///
/// - ring:  supports +, -, *, and comparison.  Integer types qualify.
/// - field: additionally supports meaningful division (no truncation).
///          Floating-point types qualify; future rational or exact
///          arithmetic types can be admitted by extending the definition.
///
/// These concepts are intentionally minimal — they test built-in type
/// categories rather than syntactic operator requirements.  The goal is
/// a single vocabulary for constraining scalar template parameters
/// across zipper and downstream libraries.

#include <concepts>
#include <type_traits>

namespace zipper::concepts {

/// A type whose values form a ring: closed under addition,
/// subtraction, and multiplication, with comparisons.
///
/// Sufficient for any computation that avoids division: dot products,
/// norms (squared), element-wise arithmetic, min/max reductions, etc.
///
/// Currently equivalent to std::is_arithmetic_v.
template <typename T>
concept ring = std::is_arithmetic_v<T>;

/// A type whose values form a field: a ring where division is
/// well-defined and produces exact or meaningful results.
///
/// Required by operations involving division: normalization, matrix
/// inverse, least-squares solvers, interpolation, centroid
/// computation, etc.
///
/// Currently satisfied by floating-point types.  Future extensions
/// could admit rational or exact arithmetic types without modifying
/// downstream constraint sites.
template <typename T>
concept field = ring<T> && std::floating_point<T>;

} // namespace zipper::concepts
#endif
