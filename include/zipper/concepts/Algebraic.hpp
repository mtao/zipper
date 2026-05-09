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
/// - ring:  supports +, -, *, and comparison.  Built-in arithmetic types
///          satisfy this via short-circuit; user-defined types can satisfy
///          it by providing the required operators.
/// - field: a ring that additionally supports meaningful division.
///          Floating-point types are pre-registered.  User-defined types
///          (e.g., rationals) opt in by specializing the HasDivision trait.

#include <concepts>
#include <type_traits>

namespace zipper::concepts {

namespace detail {

/// Opt-in trait declaring that T supports meaningful division.
///
/// This is deliberately a trait rather than a requires-expression
/// checking for operator/ because built-in integral types define
/// truncating integer division, which syntactically satisfies
/// { a / b } but does not constitute algebraic (field) division.
/// Types that wish to be recognized as fields must explicitly
/// specialize this trait to std::true_type.
///
/// Pre-registered for all floating-point types.
template <typename T>
struct HasDivision : std::false_type {};

template <std::floating_point T>
struct HasDivision<T> : std::true_type {};

} // namespace detail

/// A type whose values form a ring: closed under addition,
/// subtraction, and multiplication.
///
/// Built-in arithmetic types (int, float, etc.) satisfy the concept
/// via a short-circuit check.  User-defined types satisfy it by
/// providing the required operators (+, -, *, unary -).
///
/// Sufficient for any computation that avoids division: dot products,
/// norms (squared), element-wise arithmetic, min/max reductions, etc.
template <typename T>
concept ring =
    std::is_arithmetic_v<T>
    || requires(T a, T b) {
        { a + b } -> std::convertible_to<T>;
        { a - b } -> std::convertible_to<T>;
        { a * b } -> std::convertible_to<T>;
        { -a }    -> std::convertible_to<T>;
    };

/// A type whose values form a field: a ring where division is
/// a well-defined operation.
///
/// Division capability is declared via the HasDivision trait rather
/// than checked syntactically — see HasDivision for rationale.
///
/// Required by operations involving division: normalization, matrix
/// inverse, least-squares solvers, interpolation, centroid
/// computation, etc.
template <typename T>
concept field = ring<T> && detail::HasDivision<T>::value;

} // namespace zipper::concepts
#endif
