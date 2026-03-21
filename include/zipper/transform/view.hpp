/// @file view.hpp
/// @brief View (camera) matrix construction.
/// @ingroup transform
///
/// Provides `look_at()` which creates a right-handed view matrix, matching
/// the standard OpenGL convention where the camera looks down the negative
/// Z axis.
///
/// @code
///   using namespace zipper::transform;
///   Vector<float, 3> eye({0, 0, 3});
///   Vector<float, 3> center({0, 0, 0});
///   Vector<float, 3> up({0, 1, 0});
///   auto V = look_at(eye, center, up);
/// @endcode

#if !defined(ZIPPER_TRANSFORM_VIEW_HPP)
#define ZIPPER_TRANSFORM_VIEW_HPP

#include <zipper/Vector.hpp>
#include "AffineTransform.hpp"

namespace zipper::transform {

/// @brief Create a right-handed view matrix.
///
/// Constructs an affine transform that maps world-space coordinates into
/// eye-space, where the camera is at the origin looking down -Z.
///
/// @param eye    Camera position in world space (rank-1, extent 3).
/// @param center Target point the camera looks at (rank-1, extent 3).
/// @param up     World-space up direction (need not be normalized).
template <zipper::concepts::Vector VEye, zipper::concepts::Vector VCenter,
          zipper::concepts::Vector VUp>
    requires(VEye::extents_type::static_extent(0) == 3 &&
             VCenter::extents_type::static_extent(0) == 3 &&
             VUp::extents_type::static_extent(0) == 3)
auto look_at(const VEye& eye, const VCenter& center,
             const VUp& up) -> AffineTransform<typename VEye::value_type> {
    using T = typename VEye::value_type;

    // f = normalize(center - eye)   — forward (into the screen)
    Vector<T, 3> f = (center - eye).normalized();
    // s = normalize(f x up)         — right
    Vector<T, 3> s = f.cross(up).normalized();
    // u = s x f                     — true up
    Vector<T, 3> u = s.cross(f);

    AffineTransform<T> result;
    result(0, 0) = s(0);
    result(0, 1) = s(1);
    result(0, 2) = s(2);
    result(1, 0) = u(0);
    result(1, 1) = u(1);
    result(1, 2) = u(2);
    result(2, 0) = -f(0);
    result(2, 1) = -f(1);
    result(2, 2) = -f(2);
    result(0, 3) = -s.dot(eye);
    result(1, 3) = -u.dot(eye);
    result(2, 3) = f.dot(eye);
    result(3, 3) = T(1);
    return result;
}

} // namespace zipper::transform
#endif
