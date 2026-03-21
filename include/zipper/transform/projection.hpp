/// @file projection.hpp
/// @brief Projection matrix construction: orthographic, perspective, frustum.
/// @ingroup transform
///
/// All functions produce right-handed projection matrices with depth mapped
/// to [-1, 1] (OpenGL convention).
///
/// @code
///   using namespace zipper::transform;
///   auto P = perspective(radians(45.0f), 16.0f/9.0f, 0.1f, 100.0f);
///   auto O = ortho(-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 100.0f);
/// @endcode

#if !defined(ZIPPER_TRANSFORM_PROJECTION_HPP)
#define ZIPPER_TRANSFORM_PROJECTION_HPP

#include <cmath>
#include <concepts>

#include <zipper/Matrix.hpp>

namespace zipper::transform {

/// @brief Create a 2D orthographic projection matrix (no near/far planes).
///
/// Maps the rectangle [left, right] x [bottom, top] to [-1, 1] x [-1, 1].
/// The Z coordinate is left unchanged (identity in Z).
template <std::floating_point T>
auto ortho(T left, T right, T bottom, T top) -> Matrix<T, 4, 4> {
    Matrix<T, 4, 4> result;
    result(0, 0) = T(2) / (right - left);
    result(1, 1) = T(2) / (top - bottom);
    result(2, 2) = -T(1);
    result(0, 3) = -(right + left) / (right - left);
    result(1, 3) = -(top + bottom) / (top - bottom);
    result(3, 3) = T(1);
    return result;
}

/// @brief Create a 3D orthographic projection matrix (RH, depth [-1, 1]).
///
/// Maps the axis-aligned box [left,right] x [bottom,top] x [-zNear,-zFar]
/// to the clip cube [-1,1]^3.
template <std::floating_point T>
auto ortho(T left, T right, T bottom, T top, T zNear, T zFar)
    -> Matrix<T, 4, 4> {
    Matrix<T, 4, 4> result;
    result(0, 0) = T(2) / (right - left);
    result(1, 1) = T(2) / (top - bottom);
    result(2, 2) = -T(2) / (zFar - zNear);
    result(0, 3) = -(right + left) / (right - left);
    result(1, 3) = -(top + bottom) / (top - bottom);
    result(2, 3) = -(zFar + zNear) / (zFar - zNear);
    result(3, 3) = T(1);
    return result;
}

/// @brief Create a perspective projection from raw frustum planes (RH, depth [-1, 1]).
///
/// This is the most general perspective projection. The symmetric
/// `perspective()` function is implemented in terms of this.
template <std::floating_point T>
auto frustum(T left, T right, T bottom, T top, T zNear, T zFar)
    -> Matrix<T, 4, 4> {
    Matrix<T, 4, 4> result;
    result(0, 0) = (T(2) * zNear) / (right - left);
    result(1, 1) = (T(2) * zNear) / (top - bottom);
    result(0, 2) = (right + left) / (right - left);
    result(1, 2) = (top + bottom) / (top - bottom);
    result(2, 2) = -(zFar + zNear) / (zFar - zNear);
    result(2, 3) = -(T(2) * zFar * zNear) / (zFar - zNear);
    result(3, 2) = -T(1);
    return result;
}

/// @brief Create a symmetric perspective projection (RH, depth [-1, 1]).
///
/// @param fovy   Vertical field of view in radians.
/// @param aspect Aspect ratio (width / height).
/// @param zNear  Distance to the near clipping plane (must be positive).
/// @param zFar   Distance to the far clipping plane (must be positive).
template <std::floating_point T>
auto perspective(T fovy, T aspect, T zNear, T zFar) -> Matrix<T, 4, 4> {
    T const tanHalfFovy = std::tan(fovy / T(2));

    Matrix<T, 4, 4> result;
    result(0, 0) = T(1) / (aspect * tanHalfFovy);
    result(1, 1) = T(1) / tanHalfFovy;
    result(2, 2) = -(zFar + zNear) / (zFar - zNear);
    result(2, 3) = -(T(2) * zFar * zNear) / (zFar - zNear);
    result(3, 2) = -T(1);
    return result;
}

/// @brief Create a perspective projection with zFar at infinity (RH).
///
/// Useful for shadow mapping and stencil shadow techniques where the
/// far plane should not clip geometry.
///
/// @param fovy   Vertical field of view in radians.
/// @param aspect Aspect ratio (width / height).
/// @param zNear  Distance to the near clipping plane (must be positive).
template <std::floating_point T>
auto infinite_perspective(T fovy, T aspect, T zNear) -> Matrix<T, 4, 4> {
    T const tanHalfFovy = std::tan(fovy / T(2));

    Matrix<T, 4, 4> result;
    result(0, 0) = T(1) / (aspect * tanHalfFovy);
    result(1, 1) = T(1) / tanHalfFovy;
    result(2, 2) = -T(1);
    result(2, 3) = -T(2) * zNear;
    result(3, 2) = -T(1);
    return result;
}

} // namespace zipper::transform
#endif
