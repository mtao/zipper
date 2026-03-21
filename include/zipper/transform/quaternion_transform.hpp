/// @file quaternion_transform.hpp
/// @brief Quaternion construction and conversion functions.
/// @ingroup transform
///
/// Provides functions for creating quaternions from axis-angle
/// representations, extracting Euler angles, converting between quaternion
/// and matrix/affine representations, and performing spherical linear
/// interpolation.
///
/// All functions live in the `zipper::transform` namespace and follow
/// right-handed coordinate conventions.
///
/// @code
///   using namespace zipper::transform;
///   auto q = angle_axis(radians(90.0f), Vector<float, 3>({0, 1, 0}));
///   auto A = to_affine(q);              // quaternion -> AffineTransform
///   auto M3 = to_rotation_matrix(q);    // quaternion -> 3x3 rotation matrix
///   auto q2 = to_quaternion(A);         // affine -> quaternion
///   auto q3 = slerp(q, q2, 0.5f);      // spherical interpolation
///   auto angles = euler_angles(q);      // extract pitch, yaw, roll
/// @endcode
///
/// @see zipper::Quaternion — owning quaternion type.
/// @see zipper::transform::rotate — applies rotation via axis-angle directly.

#if !defined(ZIPPER_TRANSFORM_QUATERNION_TRANSFORM_HPP)
#define ZIPPER_TRANSFORM_QUATERNION_TRANSFORM_HPP

#include <cmath>
#include <concepts>

#include <zipper/Matrix.hpp>
#include <zipper/Quaternion.hpp>
#include <zipper/Vector.hpp>
#include "AffineTransform.hpp"

namespace zipper::transform {

/// @brief Create a quaternion from an axis-angle representation.
///
/// @param angle Rotation angle in radians.
/// @param axis  Rotation axis (rank-1, extent 3; must be normalized).
/// @return      Unit quaternion representing the rotation.
template <zipper::concepts::Vector V>
    requires(V::extents_type::static_extent(0) == 3)
auto angle_axis(typename V::value_type angle, const V& axis)
    -> Quaternion<typename V::value_type> {
    using T = typename V::value_type;
    T half = angle / T(2);
    T s = std::sin(half);
    return Quaternion<T>(std::cos(half), axis(0) * s, axis(1) * s, axis(2) * s);
}

/// @brief Extract Euler angles (pitch, yaw, roll) from a quaternion.
///
/// Returns angles in radians as (pitch, yaw, roll) corresponding to
/// rotations about the X, Y, Z axes respectively.  Uses the intrinsic
/// Tait-Bryan convention (ZYX order).
///
/// @param q Unit quaternion.
/// @return  Vector of (pitch, yaw, roll) in radians.
template <zipper::concepts::Quaternion Q>
auto euler_angles(const Q& q) -> Vector<typename Q::value_type, 3> {
    using T = typename Q::value_type;
    Vector<T, 3> result;

    // Pitch (x-axis rotation)
    T sinp = T(2) * (q.w() * q.x() + q.y() * q.z());
    T cosp = T(1) - T(2) * (q.x() * q.x() + q.y() * q.y());
    result(0) = std::atan2(sinp, cosp);

    // Yaw (y-axis rotation)
    T siny = T(2) * (q.w() * q.y() - q.z() * q.x());
    if (std::abs(siny) >= T(1)) {
        result(1) = std::copysign(std::numbers::pi_v<T> / T(2), siny);
    } else {
        result(1) = std::asin(siny);
    }

    // Roll (z-axis rotation)
    T sinr = T(2) * (q.w() * q.z() + q.x() * q.y());
    T cosr = T(1) - T(2) * (q.y() * q.y() + q.z() * q.z());
    result(2) = std::atan2(sinr, cosr);

    return result;
}

/// @brief Convert a quaternion to a 3x3 rotation matrix.
///
/// @param q Unit quaternion.
/// @return  3x3 rotation matrix.
template <zipper::concepts::Quaternion Q>
auto to_rotation_matrix(const Q& q) -> Matrix<typename Q::value_type, 3, 3> {
    using T = typename Q::value_type;
    T w = q.w(), x = q.x(), y = q.y(), z = q.z();

    T xx = x * x, yy = y * y, zz = z * z;
    T xy = x * y, xz = x * z, yz = y * z;
    T wx = w * x, wy = w * y, wz = w * z;

    Matrix<T, 3, 3> result;
    result(0, 0) = T(1) - T(2) * (yy + zz);
    result(0, 1) = T(2) * (xy - wz);
    result(0, 2) = T(2) * (xz + wy);

    result(1, 0) = T(2) * (xy + wz);
    result(1, 1) = T(1) - T(2) * (xx + zz);
    result(1, 2) = T(2) * (yz - wx);

    result(2, 0) = T(2) * (xz - wy);
    result(2, 1) = T(2) * (yz + wx);
    result(2, 2) = T(1) - T(2) * (xx + yy);

    return result;
}

/// @brief Convert a quaternion to an affine transform (4x4 rotation matrix).
///
/// The upper-left 3x3 block contains the rotation; the rest is identity.
///
/// @param q Unit quaternion.
/// @return  AffineTransform representing the rotation.
template <zipper::concepts::Quaternion Q>
auto to_affine(const Q& q) -> AffineTransform<typename Q::value_type> {
    using T = typename Q::value_type;
    auto m3 = to_rotation_matrix(q);

    AffineTransform<T> result;
    for (index_type r = 0; r < 3; ++r) {
        for (index_type c = 0; c < 3; ++c) {
            result(r, c) = m3(r, c);
        }
    }
    // result already has identity in row 3 and column 3 from default ctor
    return result;
}

/// @brief Convert a 3x3 rotation matrix to a quaternion.
///
/// Uses Shepperd's method for numerical stability.
///
/// @param m 3x3 rotation matrix.
/// @return  Unit quaternion representing the same rotation.
template <zipper::concepts::Matrix M>
    requires(M::extents_type::static_extent(0) == 3 &&
             M::extents_type::static_extent(1) == 3)
auto to_quaternion(const M& m) -> Quaternion<typename M::value_type> {
    using T = typename M::value_type;
    T trace = m(0, 0) + m(1, 1) + m(2, 2);

    if (trace > T(0)) {
        T s = T(0.5) / std::sqrt(trace + T(1));
        return Quaternion<T>(
            T(0.25) / s,
            (m(2, 1) - m(1, 2)) * s,
            (m(0, 2) - m(2, 0)) * s,
            (m(1, 0) - m(0, 1)) * s);
    } else if (m(0, 0) > m(1, 1) && m(0, 0) > m(2, 2)) {
        T s = T(2) * std::sqrt(T(1) + m(0, 0) - m(1, 1) - m(2, 2));
        return Quaternion<T>(
            (m(2, 1) - m(1, 2)) / s,
            T(0.25) * s,
            (m(0, 1) + m(1, 0)) / s,
            (m(0, 2) + m(2, 0)) / s);
    } else if (m(1, 1) > m(2, 2)) {
        T s = T(2) * std::sqrt(T(1) + m(1, 1) - m(0, 0) - m(2, 2));
        return Quaternion<T>(
            (m(0, 2) - m(2, 0)) / s,
            (m(0, 1) + m(1, 0)) / s,
            T(0.25) * s,
            (m(1, 2) + m(2, 1)) / s);
    } else {
        T s = T(2) * std::sqrt(T(1) + m(2, 2) - m(0, 0) - m(1, 1));
        return Quaternion<T>(
            (m(1, 0) - m(0, 1)) / s,
            (m(0, 2) + m(2, 0)) / s,
            (m(1, 2) + m(2, 1)) / s,
            T(0.25) * s);
    }
}

/// @brief Convert an affine transform to a quaternion.
///
/// Extracts the upper-left 3x3 rotation block and delegates to the 3x3
/// overload.
///
/// @param m AffineTransform (only the upper-left 3x3 is used).
/// @return  Unit quaternion representing the rotation.
template <concepts::AffineTransform A>
auto to_quaternion(const A& m) -> Quaternion<typename A::value_type> {
    using T = typename A::value_type;
    Matrix<T, 3, 3> m3;
    for (index_type r = 0; r < 3; ++r) {
        for (index_type c = 0; c < 3; ++c) {
            m3(r, c) = m(r, c);
        }
    }
    return to_quaternion(m3);
}

/// @brief Spherical linear interpolation between two quaternions.
///
/// Interpolates between @p q1 and @p q2 by parameter @p t in [0, 1].
/// Takes the shorter path on the unit sphere (negates q2 if the dot
/// product is negative).
///
/// @param q1 Start quaternion (unit).
/// @param q2 End quaternion (unit).
/// @param t  Interpolation parameter.
/// @return   Interpolated unit quaternion.
template <zipper::concepts::Quaternion Q1, zipper::concepts::Quaternion Q2>
auto slerp(const Q1& q1, const Q2& q2,
           typename Q1::value_type t) -> Quaternion<typename Q1::value_type> {
    using T = typename Q1::value_type;

    T cosTheta = q1.dot(q2);

    // If dot < 0, negate q2 to take the shorter path.
    Quaternion<T> q2a = q2;
    if (cosTheta < T(0)) {
        q2a = -q2;
        cosTheta = -cosTheta;
    }

    // If quaternions are very close, use linear interpolation to avoid
    // division by zero in the sin calculation.
    if (cosTheta > T(1) - std::numeric_limits<T>::epsilon()) {
        return Quaternion<T>(
            q1.w() + t * (q2a.w() - q1.w()),
            q1.x() + t * (q2a.x() - q1.x()),
            q1.y() + t * (q2a.y() - q1.y()),
            q1.z() + t * (q2a.z() - q1.z()));
    }

    T theta = std::acos(cosTheta);
    T sinTheta = std::sin(theta);
    T w1 = std::sin((T(1) - t) * theta) / sinTheta;
    T w2 = std::sin(t * theta) / sinTheta;

    return Quaternion<T>(
        w1 * q1.w() + w2 * q2a.w(),
        w1 * q1.x() + w2 * q2a.x(),
        w1 * q1.y() + w2 * q2a.y(),
        w1 * q1.z() + w2 * q2a.z());
}

} // namespace zipper::transform
#endif
