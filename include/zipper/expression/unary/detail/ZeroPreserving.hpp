#if !defined(ZIPPER_EXPRESSION_UNARY_ZEROPRESERVING_HPP)
#define ZIPPER_EXPRESSION_UNARY_ZEROPRESERVING_HPP

#include <functional>

/// @file ZeroPreserving.hpp
/// @brief Traits and concepts for marking unary/scalar ops that preserve zeros.
///
/// An operation Op is "zero-preserving" if Op(0) == 0.  When this holds, a
/// coefficient-wise expression can propagate the sparsity pattern (index set)
/// of its child unchanged.
///
/// Usage for new math functor headers:
///   1. Define your functor in zipper::expression::unary::detail.
///   2. Include this header.
///   3. Specialize is_zero_preserving_unary_op for your functor:
///        template <typename T>
///        struct zipper::expression::detail::is_zero_preserving_unary_op<
///            unary::detail::my_func<T>> : std::true_type {};

namespace zipper::expression::detail {

// ─────────────────────────────────────────────────────────────────────────────
/// @brief Trait to mark unary functors that preserve zero: Op(0) == 0.
///
/// Specialize `is_zero_preserving_unary_op<Op>` for types where this holds.
// ─────────────────────────────────────────────────────────────────────────────
template <typename Op>
struct is_zero_preserving_unary_op : std::false_type {};

// std::negate preserves zero: -0 == 0
template <typename T>
struct is_zero_preserving_unary_op<std::negate<T>> : std::true_type {};

template <typename Op>
concept ZeroPreservingUnaryOp = is_zero_preserving_unary_op<Op>::value;

// ─────────────────────────────────────────────────────────────────────────────
/// @brief Trait to mark binary scalar functors that preserve zero when
///        applied with a fixed scalar: Op(0, s) == 0 or Op(s, 0) == 0.
///
/// The template parameters are:
///   - Op: the binary functor type
///   - ScalarOnRight: true if the scalar is the second argument
// ─────────────────────────────────────────────────────────────────────────────
template <typename Op, bool ScalarOnRight>
struct is_zero_preserving_scalar_op : std::false_type {};

// multiplies: 0 * s == 0, s * 0 == 0
template <typename T, bool ScalarOnRight>
struct is_zero_preserving_scalar_op<std::multiplies<T>, ScalarOnRight>
  : std::true_type {};

// divides: 0 / s == 0 (scalar on right only)
template <typename T>
struct is_zero_preserving_scalar_op<std::divides<T>, /*ScalarOnRight=*/true>
  : std::true_type {};

template <typename Op, bool ScalarOnRight>
concept ZeroPreservingScalarOp =
    is_zero_preserving_scalar_op<Op, ScalarOnRight>::value;

} // namespace zipper::expression::detail
#endif
