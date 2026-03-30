/// @file all.hpp
/// @brief Umbrella header for all transform functions.
/// @ingroup transform
///
/// Includes all transform sub-headers: common utilities, projection matrices,
/// view matrices, model transformations, geometric functions, coordinate
/// mapping, quaternion operations, and the unified Transform type.
///
/// @note This file was renamed from `transform.hpp` to `all.hpp` to avoid a
///       case collision with `Transform.hpp` on case-insensitive filesystems
///       (macOS).
///
/// @code
///   #include <zipper/transform/all.hpp>
///   using namespace zipper::transform;
/// @endcode

#if !defined(ZIPPER_TRANSFORM_ALL_HPP)
#define ZIPPER_TRANSFORM_ALL_HPP

#include "AxisAngleRotation.hpp"
#include "Rotation.hpp"
#include "Scaling.hpp"
#include "Transform.hpp"
#include "Translation.hpp"
#include "common.hpp"
#include "coordinate.hpp"
#include "decompose.hpp"
#include "geometric.hpp"
#include "model.hpp"
#include "projection.hpp"
#include "quaternion_transform.hpp"
#include "transform_compose.hpp"
#include "view.hpp"

#endif
