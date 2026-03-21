/// @file model.hpp
/// @brief Model transformation factories: translation, rotation, scaling.
/// @ingroup transform
///
/// These functions produce standalone specialized Transform objects
/// with minimal storage.  Users compose them via `operator*`:
///
/// @code
///   using namespace zipper::transform;
///   Vector<float, 3> v({1, 2, 3});
///   auto T = translation(v);                             // Translation<float, 3>
///   auto R = rotation(radians(45.0f), Vector<float,3>({0,1,0})); // Rotation<float, 3>
///   auto S = scaling(Vector<float, 3>({2, 2, 2}));       // Scaling<float, 3>
///   auto model = T * R * S;                              // AffineTransform (promoted)
///
///   // 2D:
///   auto T2 = translation(Vector<float, 2>({1, 2}));     // Translation<float, 2>
///   auto S2 = scaling(Vector<float, 2>({2, 3}));         // Scaling<float, 2>
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
#include "Translation.hpp"
#include "Rotation.hpp"
#include "Scaling.hpp"
#include "transform_compose.hpp"

namespace zipper::transform {

/// @brief Create a translation transform (dimension-agnostic).
///
/// Returns a Translation with minimal storage (Vector<T, D>).
///
/// @param v  Translation vector (rank-1, extent D).
/// @return   Translation<T, D>.
template <zipper::concepts::Vector V>
auto translation(const V& v) {
    using T = typename std::decay_t<V>::value_type;
    constexpr index_type D = std::decay_t<V>::extents_type::static_extent(0);
    return Translation<T, D>(v);
}

/// @brief Create a rotation transform (3D only, axis-angle).
///
/// Returns a Rotation with minimal storage (Matrix<T, 3, 3>).
///
/// @param angle Rotation angle in radians.
/// @param axis  Rotation axis (rank-1, extent 3; need not be normalized).
/// @return      Rotation<T, 3>.
template <std::floating_point T, zipper::concepts::Vector V>
    requires(std::decay_t<V>::extents_type::static_extent(0) == 3)
auto rotation(T angle, const V& axis) {
    T const c = std::cos(angle);
    T const s = std::sin(angle);
    Vector<T, 3> a = Vector<T, 3>(axis).normalized();
    Vector<T, 3> t = a * (T(1) - c); // temp = axis * (1 - cos)

    // Rodrigues' rotation formula entries
    Rotation<T, 3> result;
    result(0, 0) = c + t(0) * a(0);
    result(0, 1) = t(0) * a(1) - s * a(2);
    result(0, 2) = t(0) * a(2) + s * a(1);

    result(1, 0) = t(1) * a(0) + s * a(2);
    result(1, 1) = c + t(1) * a(1);
    result(1, 2) = t(1) * a(2) - s * a(0);

    result(2, 0) = t(2) * a(0) - s * a(1);
    result(2, 1) = t(2) * a(1) + s * a(0);
    result(2, 2) = c + t(2) * a(2);

    return result;
}

/// @brief Create a non-uniform scaling transform (dimension-agnostic).
///
/// Returns a Scaling with minimal storage (Vector<T, D>).
///
/// @param v  Scale factors (rank-1, extent D).
/// @return   Scaling<T, D>.
template <zipper::concepts::Vector V>
auto scaling(const V& v) {
    using T = typename std::decay_t<V>::value_type;
    constexpr index_type D = std::decay_t<V>::extents_type::static_extent(0);
    return Scaling<T, D>(v);
}

} // namespace zipper::transform
#endif
