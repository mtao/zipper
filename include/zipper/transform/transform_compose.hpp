/// @file transform_compose.hpp
/// @brief Generic composition operators for mixed transform types.
/// @ingroup transform
///
/// Provides fallback `operator*` overloads that convert specialized
/// transform types (Translation, Scaling, Rotation, AxisAngleRotation)
/// to their full (D+1)x(D+1) matrix form for composition with other
/// transform types.
///
/// Same-type compositions (e.g. Translation * Translation) are handled
/// by more specific overloads in each type's header and do NOT come here.

#if !defined(ZIPPER_TRANSFORM_TRANSFORM_COMPOSE_HPP)
#define ZIPPER_TRANSFORM_TRANSFORM_COMPOSE_HPP

#include "Translation.hpp"
#include "Scaling.hpp"
#include "Rotation.hpp"
#include "AxisAngleRotation.hpp"
#include "Transform.hpp"

namespace zipper::transform {

// ============================================================================
// Generic fallback: specialized * MatrixTransform
// ============================================================================

/// @brief Specialized transform * MatrixTransform → promoted Transform.
///
/// Converts the specialized type to a full matrix transform, then delegates
/// to the existing MatrixTransform * MatrixTransform operator.
template <concepts::Transform A, concepts::MatrixTransform B>
    requires(!concepts::MatrixTransform<A> && std::decay_t<A>::dim == std::decay_t<B>::dim)
auto operator*(const A& lhs, const B& rhs) {
    return lhs.to_transform() * rhs;
}

/// @brief MatrixTransform * Specialized transform → promoted Transform.
template <concepts::MatrixTransform A, concepts::Transform B>
    requires(!concepts::MatrixTransform<B> && std::decay_t<A>::dim == std::decay_t<B>::dim)
auto operator*(const A& lhs, const B& rhs) {
    return lhs * rhs.to_transform();
}

// ============================================================================
// Generic fallback: specialized * specialized (different types)
// ============================================================================

/// @brief Specialized * Specialized (different types) → promoted Transform.
///
/// Both operands are converted to full matrix transforms.  Same-type
/// compositions (Translation*Translation, Rotation*Rotation, etc.) are
/// handled by more constrained overloads in each type's header.
template <concepts::Transform A, concepts::Transform B>
    requires(!concepts::MatrixTransform<A> && !concepts::MatrixTransform<B> &&
             std::decay_t<A>::dim == std::decay_t<B>::dim &&
             !std::is_same_v<std::decay_t<A>, std::decay_t<B>>)
auto operator*(const A& lhs, const B& rhs) {
    return lhs.to_transform() * rhs.to_transform();
}

} // namespace zipper::transform
#endif
