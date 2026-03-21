/// @file common.hpp
/// @brief Common math utilities for graphics: angle conversion.
/// @ingroup transform
///
/// Provides angle conversion functions for the transform library.
/// All functions live in the `zipper::transform` namespace.
///
/// @code
///   using namespace zipper::transform;
///   float r = radians(90.0f);          // pi/2
///   float d = degrees(3.14159f);       // ~180
/// @endcode

#if !defined(ZIPPER_TRANSFORM_COMMON_HPP)
#define ZIPPER_TRANSFORM_COMMON_HPP

#include <cmath>
#include <concepts>
#include <numbers>

namespace zipper::transform {

/// @brief Convert degrees to radians.
template <std::floating_point T>
constexpr auto radians(T degrees) -> T {
    return degrees * static_cast<T>(std::numbers::pi_v<T> / T(180));
}

/// @brief Convert radians to degrees.
template <std::floating_point T>
constexpr auto degrees(T radians) -> T {
    return radians * static_cast<T>(T(180) / std::numbers::pi_v<T>);
}

} // namespace zipper::transform
#endif
