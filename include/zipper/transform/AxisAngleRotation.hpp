/// @file AxisAngleRotation.hpp
/// @brief Specialized 3D axis-angle rotation with minimal storage.
/// @ingroup transform
///
/// `AxisAngleRotation<T>` stores an angle and a normalized axis vector,
/// rather than a full rotation matrix.  It satisfies `concepts::Isometry`
/// and can be composed with other transform types via `operator*`.
///
/// For composition, it converts to a `Rotation<T, 3>` since axis-angle
/// representation is not closed under composition.
///
/// @code
///   using namespace zipper::transform;
///   AxisAngleRotation<float> r(radians(90.0f), {0, 1, 0});
///   Vector<float, 3> p({1, 0, 0});
///   Vector<float, 3> q = r * p;     // rotated point
///   auto inv = r.inverse();         // negated angle
///   auto rot = r.to_rotation();     // Rotation<float, 3>
/// @endcode

#if !defined(ZIPPER_TRANSFORM_AXISANGLEROTATION_HPP)
#define ZIPPER_TRANSFORM_AXISANGLEROTATION_HPP

#include "Rotation.hpp"

namespace zipper::transform {

/// @brief 3D axis-angle rotation with minimal storage (angle + axis).
template <typename T>
class AxisAngleRotation {
  public:
    using value_type = T;
    static constexpr index_type dim = 3;
    static constexpr index_type D = 3;
    static constexpr TransformMode mode = TransformMode::Isometry;
    static constexpr index_type H = 4;

    /// @brief Default-construct to identity (zero angle).
    AxisAngleRotation() : m_angle(T(0)), m_axis() {
        m_axis(0) = T(1);  // arbitrary unit axis
    }

    /// @brief Construct from angle and axis vector (axis is normalized).
    template <zipper::concepts::Vector V>
        requires(std::decay_t<V>::extents_type::static_extent(0) == 3)
    AxisAngleRotation(T angle, const V& axis) : m_angle(angle), m_axis() {
        // Normalize the axis
        T len = T(0);
        for (index_type i = 0; i < 3; ++i) {
            m_axis(i) = static_cast<T>(axis(i));
            len += m_axis(i) * m_axis(i);
        }
        len = std::sqrt(len);
        for (index_type i = 0; i < 3; ++i) {
            m_axis(i) /= len;
        }
    }

    /// @brief Construct from angle and initializer list for axis.
    AxisAngleRotation(T angle, std::initializer_list<T> axis_init)
        : m_angle(angle), m_axis(axis_init) {
        // Normalize
        T len = T(0);
        for (index_type i = 0; i < 3; ++i) {
            len += m_axis(i) * m_axis(i);
        }
        len = std::sqrt(len);
        for (index_type i = 0; i < 3; ++i) {
            m_axis(i) /= len;
        }
    }

    /// @brief Get the rotation angle.
    T angle() const { return m_angle; }
    T& angle() { return m_angle; }

    /// @brief Get the rotation axis (normalized).
    auto const& axis() const { return m_axis; }
    auto& axis() { return m_axis; }

    /// @brief Return a zero translation vector (concept API).
    ///
    /// AxisAngleRotation has no translation component.
    auto translation() const { return zipper::Vector<T, 3>{}; }

    /// @brief Return the 3x3 rotation matrix as the linear part (concept API).
    ///
    /// Computed via Rodrigues' formula (same as to_rotation().matrix()).
    auto linear() const { return to_rotation().matrix(); }

    /// @brief Inverse: negate the angle.
    AxisAngleRotation inverse() const {
        AxisAngleRotation result;
        result.m_angle = -m_angle;
        result.m_axis = m_axis;
        return result;
    }

    /// @brief Convert to a Rotation<T, 3> (3x3 rotation matrix).
    ///
    /// Uses Rodrigues' rotation formula.
    Rotation<T, 3> to_rotation() const {
        T const c = std::cos(m_angle);
        T const s = std::sin(m_angle);
        T const t = T(1) - c;

        T const& x = m_axis(0);
        T const& y = m_axis(1);
        T const& z = m_axis(2);

        Rotation<T, 3> result;
        result(0, 0) = c + t * x * x;
        result(0, 1) = t * x * y - s * z;
        result(0, 2) = t * x * z + s * y;

        result(1, 0) = t * y * x + s * z;
        result(1, 1) = c + t * y * y;
        result(1, 2) = t * y * z - s * x;

        result(2, 0) = t * z * x - s * y;
        result(2, 1) = t * z * y + s * x;
        result(2, 2) = c + t * z * z;

        return result;
    }

    /// @brief Convert to a full (D+1)x(D+1) Isometry transform.
    auto to_transform() const -> Transform<T, 3, TransformMode::Isometry> {
        return to_rotation().to_transform();
    }

    /// @brief Convert to an owning 4x4 matrix.
    auto to_matrix() const -> zipper::Matrix<T, 4, 4> {
        return to_transform().to_matrix();
    }

  private:
    T m_angle;
    zipper::Vector<T, 3> m_axis;
};

// Register AxisAngleRotation as a Transform type.
namespace detail {
template <typename T>
struct IsTransform<AxisAngleRotation<T>> : std::true_type {};
} // namespace detail

// ============================================================================
// AxisAngleRotation composition operators
// ============================================================================

/// @brief AxisAngleRotation * AxisAngleRotation → Rotation (not closed in axis-angle).
template <typename T>
auto operator*(const AxisAngleRotation<T>& lhs, const AxisAngleRotation<T>& rhs) {
    return lhs.to_rotation() * rhs.to_rotation();
}

/// @brief AxisAngleRotation * Rotation → Rotation.
template <typename T>
auto operator*(const AxisAngleRotation<T>& lhs, const Rotation<T, 3>& rhs) {
    return lhs.to_rotation() * rhs;
}

/// @brief Rotation * AxisAngleRotation → Rotation.
template <typename T>
auto operator*(const Rotation<T, 3>& lhs, const AxisAngleRotation<T>& rhs) {
    return lhs * rhs.to_rotation();
}

/// @brief AxisAngleRotation * Vector<3> → rotated point.
///
/// Applies Rodrigues' formula directly without building the full matrix:
///   v' = v*cos(θ) + (axis × v)*sin(θ) + axis*(axis·v)*(1-cos(θ))
template <typename T, zipper::concepts::Vector V>
    requires(std::decay_t<V>::extents_type::static_extent(0) == 3)
auto operator*(const AxisAngleRotation<T>& lhs, const V& rhs) {
    T const c = std::cos(lhs.angle());
    T const s = std::sin(lhs.angle());
    auto const& a = lhs.axis();

    // dot = axis · v
    T dot = T(0);
    for (index_type i = 0; i < 3; ++i) {
        dot += a(i) * rhs(i);
    }

    // cross = axis × v
    zipper::Vector<T, 3> cross;
    cross(0) = a(1) * rhs(2) - a(2) * rhs(1);
    cross(1) = a(2) * rhs(0) - a(0) * rhs(2);
    cross(2) = a(0) * rhs(1) - a(1) * rhs(0);

    // Rodrigues: v*cos + cross*sin + axis*(dot)*(1-cos)
    zipper::Vector<T, 3> result;
    for (index_type i = 0; i < 3; ++i) {
        result(i) = rhs(i) * c + cross(i) * s + a(i) * dot * (T(1) - c);
    }
    return result;
}

} // namespace zipper::transform
#endif
