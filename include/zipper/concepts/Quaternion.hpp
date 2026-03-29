/// @file Quaternion.hpp
/// @brief Concept definition for quaternion types.
/// @ingroup concepts
///
/// Defines the `Quaternion` concept via an `IsQuaternion` trait, following
/// the same registration pattern as `IsVector`, `IsMatrix`, etc.

#if !defined(ZIPPER_CONCEPTS_QUATERNION_HPP)
#define ZIPPER_CONCEPTS_QUATERNION_HPP
#include <concepts>
#include <type_traits>

#include "Expression.hpp"

namespace zipper {
template <concepts::Expression T>
  requires(concepts::QualifiedRankedExpression<T, 1>)
class QuaternionBase;
template <typename T> class Quaternion;
} // namespace zipper
namespace zipper::concepts {
namespace detail {
template <typename> struct IsQuaternion : std::false_type {};

} // namespace detail

template <typename T>
concept Quaternion = detail::IsQuaternion<std::decay_t<T>>::value;

} // namespace zipper::concepts
#endif
