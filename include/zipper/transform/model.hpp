/// @file model.hpp
/// @brief Model transformation factories: translation, rotation, scaling.
/// @ingroup transform
///
/// These functions produce standalone Transform objects representing
/// individual transformations.  Users compose them via `operator*`:
///
/// @code
///   using namespace zipper::transform;
///   Vector<float, 3> v({1, 2, 3});
///   auto T = translation(v);                             // Isometry<float, 3>
///   auto R = rotation(radians(45.0f), Vector<float,3>({0,1,0})); // Isometry<float, 3>
///   auto S = scaling(Vector<float, 3>({2, 2, 2}));       // AffineTransform<float, 3>
///   auto model = T * R * S;                            // AffineTransform (mode promotion)
///
///   // 2D:
///   auto T2 = translation(Vector<float, 2>({1, 2}));     // Isometry<float, 2>
///   auto S2 = scaling(Vector<float, 2>({2, 3}));         // AffineTransform<float, 2>
/// @endcode
///
/// `translation` and `scaling` are dimension-agnostic.  `rotation` (axis-angle)
/// is inherently 3D.

#if !defined(ZIPPER_TRANSFORM_MODEL_HPP)
#define ZIPPER_TRANSFORM_MODEL_HPP

#include <cmath>
#include <concepts>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include "Transform.hpp"

namespace zipper::transform {

/// @brief Create a translation transform (dimension-agnostic).
///
/// Returns an Isometry (pure translation is a rigid-body transform).
///
/// @param v  Translation vector (rank-1, extent D).
/// @return   Isometry<T, D> with translation set to v.
template <zipper::concepts::Vector V>
auto translation(const V& v) {
    using T = typename std::decay_t<V>::value_type;
    constexpr index_type D = std::decay_t<V>::extents_type::static_extent(0);

    Isometry<T, D> result;  // identity
    for (index_type i = 0; i < D; ++i) {
        result(i, D) = v(i);
    }
    return result;
}

/// @brief Create a rotation transform (3D only, axis-angle).
///
/// Returns an Isometry (rotation is a rigid-body transform).
///
/// @param angle Rotation angle in radians.
/// @param axis  Rotation axis (rank-1, extent 3; need not be normalized).
/// @return      Isometry<T, 3> representing the rotation.
template <std::floating_point T, zipper::concepts::Vector V>
    requires(std::decay_t<V>::extents_type::static_extent(0) == 3)
auto rotation(T angle, const V& axis) {
    T const c = std::cos(angle);
    T const s = std::sin(angle);
    Vector<T, 3> a = Vector<T, 3>(axis).normalized();
    Vector<T, 3> t = a * (T(1) - c); // temp = axis * (1 - cos)

    // Rodrigues' rotation formula entries
    T rot00 = c + t(0) * a(0);
    T rot01 = t(0) * a(1) - s * a(2);
    T rot02 = t(0) * a(2) + s * a(1);

    T rot10 = t(1) * a(0) + s * a(2);
    T rot11 = c + t(1) * a(1);
    T rot12 = t(1) * a(2) - s * a(0);

    T rot20 = t(2) * a(0) - s * a(1);
    T rot21 = t(2) * a(1) + s * a(0);
    T rot22 = c + t(2) * a(2);

    Isometry<T, 3> result;
    result(0, 0) = rot00; result(0, 1) = rot01; result(0, 2) = rot02;
    result(1, 0) = rot10; result(1, 1) = rot11; result(1, 2) = rot12;
    result(2, 0) = rot20; result(2, 1) = rot21; result(2, 2) = rot22;
    return result;
}

/// @brief Create a non-uniform scaling transform (dimension-agnostic).
///
/// Returns an AffineTransform (scaling breaks orthogonality).
///
/// @param v  Scale factors (rank-1, extent D).
/// @return   AffineTransform<T, D> with diagonal set to v.
template <zipper::concepts::Vector V>
auto scaling(const V& v) {
    using T = typename std::decay_t<V>::value_type;
    constexpr index_type D = std::decay_t<V>::extents_type::static_extent(0);

    AffineTransform<T, D> result;  // identity
    for (index_type i = 0; i < D; ++i) {
        result(i, i) = v(i);
    }
    return result;
}

} // namespace zipper::transform
#endif
