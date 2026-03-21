/// @file transform.hpp
/// @brief Umbrella header for all transform functions.
/// @ingroup transform
///
/// Includes all transform sub-headers: common utilities, projection matrices,
/// view matrices, model transformations, geometric functions, coordinate
/// mapping, quaternion operations, and the unified Transform type.
///
/// @code
///   #include <zipper/transform/transform.hpp>
///   using namespace zipper::transform;
/// @endcode

#if !defined(ZIPPER_TRANSFORM_UMBRELLA_HPP)
#define ZIPPER_TRANSFORM_UMBRELLA_HPP

#include "Transform.hpp"
#include "common.hpp"
#include "coordinate.hpp"
#include "geometric.hpp"
#include "model.hpp"
#include "projection.hpp"
#include "quaternion_transform.hpp"
#include "view.hpp"

#endif
