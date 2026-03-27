/// @file geometric.hpp
/// @brief Geometric vector functions: reflect, refract, face_forward, distance.
/// @ingroup transform
///
/// These functions operate on vectors of any dimension and implement the
/// standard geometric operations.
///
/// @code
///   using namespace zipper::transform;
///   Vector<float, 3> I({1, -1, 0});
///   Vector<float, 3> N({0, 1, 0});
///   auto R = reflect(I, N);     // {1, 1, 0}
///   float d = distance(I, N);   // sqrt(5)
/// @endcode

#if !defined(ZIPPER_TRANSFORM_GEOMETRIC_HPP)
#define ZIPPER_TRANSFORM_GEOMETRIC_HPP

#include <cmath>

#include <zipper/Vector.hpp>
#include <zipper/concepts/Vector.hpp>

namespace zipper::transform {

/// @brief Compute the distance between two points.
///
/// Equivalent to (p1 - p0).norm().
template <zipper::concepts::Vector V>
auto distance(const V& p0, const V& p1) -> typename V::value_type {
    return Vector(p1 - p0).norm();
}

/// @brief Reflect incident vector I around normal N.
///
/// Formula: I - 2 * dot(N, I) * N.
/// N must be normalized.
template <zipper::concepts::Vector V>
auto reflect(const V& I, const V& N) {
    using T = typename V::value_type;
    auto d = N.dot(I);
    return Vector(I - N * (T(2) * d));
}

/// @brief Compute refraction of incident vector I through a surface.
///
/// @param I   Normalized incident vector.
/// @param N   Normalized surface normal.
/// @param eta Ratio of indices of refraction (n1/n2).
/// @return    Refracted vector, or zero vector if total internal reflection occurs.
template <zipper::concepts::Vector V>
auto refract(const V& I, const V& N, typename V::value_type eta) {
    using T = typename V::value_type;
    constexpr index_type Size = V::extents_type::static_extent(0);

    T d = N.dot(I);
    T k = T(1) - eta * eta * (T(1) - d * d);
    if (k < T(0)) {
        Vector<T, Size> zero;
        return zero;
    }
    return Vector(I * eta - N * (eta * d + std::sqrt(k)));
}

/// @brief Orient normal N to face the viewer.
///
/// Returns N if dot(Nref, I) < 0, otherwise -N.
template <zipper::concepts::Vector V>
auto face_forward(const V& N, const V& I, const V& Nref) {
    if (Nref.dot(I) < typename V::value_type(0)) {
        return Vector(N);
    } else {
        return Vector(-N);
    }
}

} // namespace zipper::transform
#endif
