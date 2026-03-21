/// @file model.hpp
/// @brief Model transformation functions: translate, rotate, scale.
/// @ingroup transform
///
/// These functions apply transformations to an existing 4x4 matrix by
/// post-multiplying the appropriate transformation matrix: result = m * T,
/// matching GLM's convention.
///
/// @code
///   using namespace zipper::transform;
///   AffineTransform<float> xform;
///   xform = translate(xform, Vector<float, 3>({1, 2, 3}));
///   xform = rotate(xform, radians(45.0f), Vector<float, 3>({0, 1, 0}));
///   xform = scale(xform, Vector<float, 3>({2, 2, 2}));
/// @endcode

#if !defined(ZIPPER_TRANSFORM_MODEL_HPP)
#define ZIPPER_TRANSFORM_MODEL_HPP

#include <cmath>
#include <concepts>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include "AffineTransform.hpp"

namespace zipper::transform {

/// @brief Apply a translation to an affine transform.
///
/// Equivalent to m * T, where T is a translation matrix built from v.
/// This is more efficient than building T and multiplying, as only the
/// last column needs updating.
///
/// @param m  Affine transform to modify.
/// @param v  Translation vector (rank-1, extent 3).
template <concepts::AffineTransform M, zipper::concepts::Vector V>
    requires(V::extents_type::static_extent(0) == 3)
auto translate(const M& m, const V& v)
    -> AffineTransform<typename M::value_type> {
    using T = typename M::value_type;
    AffineTransform<T> result(m);
    // result[3] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3]
    // In row-major (row, col) notation, column 3 of result =
    //   col(0)*v(0) + col(1)*v(1) + col(2)*v(2) + col(3)
    for (index_type r = 0; r < 4; ++r) {
        result(r, 3) = m(r, 0) * v(0) + m(r, 1) * v(1) + m(r, 2) * v(2) + m(r, 3);
    }
    return result;
}

/// @brief Apply a rotation to an affine transform.
///
/// Rotates by @p angle radians around the given @p axis.
/// Equivalent to m * R, where R is the rotation matrix.
///
/// @param m     The affine transform to modify.
/// @param angle Rotation angle in radians.
/// @param axis  Rotation axis (rank-1, extent 3; need not be normalized).
template <concepts::AffineTransform M, zipper::concepts::Vector V>
    requires(V::extents_type::static_extent(0) == 3)
auto rotate(const M& m, typename M::value_type angle, const V& axis)
    -> AffineTransform<typename M::value_type> {
    using T = typename M::value_type;
    T const c = std::cos(angle);
    T const s = std::sin(angle);
    Vector<T, 3> a = Vector<T, 3>(axis).normalized();
    Vector<T, 3> t = a * (T(1) - c); // temp = axis * (1 - cos)

    // Build the 3x3 rotation matrix entries (row-major convention).
    // These are the entries of Rot such that result = m * Rot.
    T rot00 = c + t(0) * a(0);
    T rot01 = t(0) * a(1) - s * a(2);
    T rot02 = t(0) * a(2) + s * a(1);

    T rot10 = t(1) * a(0) + s * a(2);
    T rot11 = c + t(1) * a(1);
    T rot12 = t(1) * a(2) - s * a(0);

    T rot20 = t(2) * a(0) - s * a(1);
    T rot21 = t(2) * a(1) + s * a(0);
    T rot22 = c + t(2) * a(2);

    // result = m * Rot (only the first 3 columns are affected)
    AffineTransform<T> result;
    for (index_type r = 0; r < 4; ++r) {
        result(r, 0) = m(r, 0) * rot00 + m(r, 1) * rot10 + m(r, 2) * rot20;
        result(r, 1) = m(r, 0) * rot01 + m(r, 1) * rot11 + m(r, 2) * rot21;
        result(r, 2) = m(r, 0) * rot02 + m(r, 1) * rot12 + m(r, 2) * rot22;
        result(r, 3) = m(r, 3);
    }
    return result;
}

/// @brief Apply a non-uniform scale to an affine transform.
///
/// Equivalent to m * S, where S is a diagonal scaling matrix built from v.
///
/// @param m  Affine transform to modify.
/// @param v  Scale factors (rank-1, extent 3).
template <concepts::AffineTransform M, zipper::concepts::Vector V>
    requires(V::extents_type::static_extent(0) == 3)
auto scale(const M& m, const V& v)
    -> AffineTransform<typename M::value_type> {
    using T = typename M::value_type;
    AffineTransform<T> result;
    for (index_type r = 0; r < 4; ++r) {
        result(r, 0) = m(r, 0) * v(0);
        result(r, 1) = m(r, 1) * v(1);
        result(r, 2) = m(r, 2) * v(2);
        result(r, 3) = m(r, 3);
    }
    return result;
}

} // namespace zipper::transform
#endif
