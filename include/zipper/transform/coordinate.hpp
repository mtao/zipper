/// @file coordinate.hpp
/// @brief Coordinate mapping: project and unproject (world <-> screen).
/// @ingroup transform
///
/// These functions replicate `gluProject` and `gluUnProject` for mapping
/// between object-space and window-space coordinates.
///
/// @code
///   using namespace zipper::transform;
///   auto win = project(obj, modelView, projection, viewport);
///   auto obj2 = unproject(win, modelView, projection, viewport);
/// @endcode

#if !defined(ZIPPER_TRANSFORM_COORDINATE_HPP)
#define ZIPPER_TRANSFORM_COORDINATE_HPP

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/inverse.hpp>

namespace zipper::transform {

/// @brief Project a 3D object coordinate to 2D window coordinates.
///
/// Replicates `gluProject`. Applies model-view and projection transforms,
/// performs perspective division, then maps to the viewport.
///
/// @param obj      Object-space 3D coordinate (rank-1, extent 3).
/// @param model    Model-view matrix (rank-2, 4x4).
/// @param proj     Projection matrix (rank-2, 4x4).
/// @param viewport Viewport parameters as (x, y, width, height) (rank-1, extent 4).
/// @return         Window-space coordinate (x, y, depth).
template <concepts::Vector VObj, concepts::Matrix MMod, concepts::Matrix MProj,
          concepts::Vector VVp>
    requires(VObj::extents_type::static_extent(0) == 3 &&
             MMod::extents_type::static_extent(0) == 4 &&
             MMod::extents_type::static_extent(1) == 4 &&
             MProj::extents_type::static_extent(0) == 4 &&
             MProj::extents_type::static_extent(1) == 4 &&
             VVp::extents_type::static_extent(0) == 4)
auto project(const VObj& obj, const MMod& model, const MProj& proj,
             const VVp& viewport)
    -> Vector<typename VObj::value_type, 3> {
    using T = typename VObj::value_type;

    // Transform to clip space: tmp = proj * model * (obj, 1)
    Vector<T, 4> tmp;
    tmp(0) = obj(0);
    tmp(1) = obj(1);
    tmp(2) = obj(2);
    tmp(3) = T(1);

    tmp = Vector<T, 4>(Matrix<T, 4, 4>(model) * tmp);
    tmp = Vector<T, 4>(Matrix<T, 4, 4>(proj) * tmp);

    // Perspective division: NDC in [-1, 1]
    tmp = tmp / tmp(3);

    // Map to window coordinates
    // NDC [-1,1] -> [0,1] -> viewport
    Vector<T, 3> result;
    result(0) = viewport(0) + viewport(2) * (tmp(0) + T(1)) / T(2);
    result(1) = viewport(1) + viewport(3) * (tmp(1) + T(1)) / T(2);
    result(2) = (tmp(2) + T(1)) / T(2);  // depth [0, 1]
    return result;
}

/// @brief Un-project a 2D window coordinate back to 3D object space.
///
/// Replicates `gluUnProject`. Inverts the projection pipeline to recover
/// the object-space coordinate from window coordinates.
///
/// @param win      Window-space coordinate (x, y, depth) (rank-1, extent 3).
/// @param model    Model-view matrix (rank-2, 4x4).
/// @param proj     Projection matrix (rank-2, 4x4).
/// @param viewport Viewport parameters as (x, y, width, height) (rank-1, extent 4).
/// @return         Object-space 3D coordinate.
template <concepts::Vector VWin, concepts::Matrix MMod, concepts::Matrix MProj,
          concepts::Vector VVp>
    requires(VWin::extents_type::static_extent(0) == 3 &&
             MMod::extents_type::static_extent(0) == 4 &&
             MMod::extents_type::static_extent(1) == 4 &&
             MProj::extents_type::static_extent(0) == 4 &&
             MProj::extents_type::static_extent(1) == 4 &&
             VVp::extents_type::static_extent(0) == 4)
auto unproject(const VWin& win, const MMod& model, const MProj& proj,
               const VVp& viewport)
    -> Vector<typename VWin::value_type, 3> {
    using T = typename VWin::value_type;

    Matrix<T, 4, 4> inv = zipper::utils::inverse(
        Matrix<T, 4, 4>(Matrix<T, 4, 4>(proj) * Matrix<T, 4, 4>(model)));

    // Map window coordinates to NDC [-1, 1]
    Vector<T, 4> tmp;
    tmp(0) = (win(0) - viewport(0)) / viewport(2) * T(2) - T(1);
    tmp(1) = (win(1) - viewport(1)) / viewport(3) * T(2) - T(1);
    tmp(2) = win(2) * T(2) - T(1);  // depth [0,1] -> [-1,1]
    tmp(3) = T(1);

    // Transform back to object space
    Vector<T, 4> obj_result = Vector<T, 4>(inv * tmp);

    Vector<T, 3> result;
    result(0) = obj_result(0) / obj_result(3);
    result(1) = obj_result(1) / obj_result(3);
    result(2) = obj_result(2) / obj_result(3);
    return result;
}

} // namespace zipper::transform
#endif
