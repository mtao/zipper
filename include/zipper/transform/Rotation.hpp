/// @file Rotation.hpp
/// @brief Specialized rotation transform with minimal storage.
/// @ingroup transform
///
/// `Rotation<T, D>` stores a DxD orthogonal rotation matrix,
/// rather than a full (D+1)x(D+1) matrix.  It satisfies `concepts::Isometry`
/// and can be composed with other transform types via `operator*`.
///
/// @code
///   using namespace zipper::transform;
///   Rotation<float, 3> r(rotation_matrix);
///   Vector<float, 3> p({1, 0, 0});
///   Vector<float, 3> q = r * p;     // rotated point
///   auto inv = r.inverse();         // transpose (orthogonal)
///   auto r2 = r * r;                // composed rotation
/// @endcode

#if !defined(ZIPPER_TRANSFORM_ROTATION_HPP)
#define ZIPPER_TRANSFORM_ROTATION_HPP

#include "detail/TransformBase.hpp"

namespace zipper::transform {

/// @brief Rotation transform with minimal storage (Matrix<T, D, D>).
template <typename T, index_type D = 3>
class Rotation {
  public:
    using value_type = T;
    static constexpr index_type dim = D;
    static constexpr TransformMode mode = TransformMode::Isometry;
    static constexpr index_type H = D + 1;

    /// @brief Default-construct to identity rotation.
    Rotation() : m_mat() {
        for (index_type i = 0; i < D; ++i) {
            m_mat(i, i) = T(1);
        }
    }

    /// @brief Construct from a DxD matrix.
    template <zipper::concepts::Matrix M>
        requires(std::decay_t<M>::extents_type::static_extent(0) == D &&
                 std::decay_t<M>::extents_type::static_extent(1) == D)
    Rotation(const M& m) : m_mat() {
        for (index_type r = 0; r < D; ++r) {
            for (index_type c = 0; c < D; ++c) {
                m_mat(r, c) = static_cast<T>(m(r, c));
            }
        }
    }

    /// @brief Access the rotation matrix (const).
    auto const& matrix() const { return m_mat; }
    /// @brief Access the rotation matrix (mutable).
    auto& matrix() { return m_mat; }

    /// @brief Return a zero translation vector (concept API).
    ///
    /// Rotation has no translation component.
    auto translation() const { return zipper::Vector<T, D>{}; }

    /// @brief Return the rotation matrix as the linear part (concept API).
    auto const& linear() const { return m_mat; }
    /// @brief Return the rotation matrix as the linear part (mutable, concept API).
    auto& linear() { return m_mat; }

    /// @brief Element access on the DxD rotation matrix.
    T operator()(index_type r, index_type c) const { return m_mat(r, c); }
    T& operator()(index_type r, index_type c) { return m_mat(r, c); }

    /// @brief Inverse: transpose (orthogonal matrix).
    Rotation inverse() const {
        Rotation result;
        for (index_type r = 0; r < D; ++r) {
            for (index_type c = 0; c < D; ++c) {
                result.m_mat(r, c) = m_mat(c, r);
            }
        }
        return result;
    }

    /// @brief Convert to a full (D+1)x(D+1) Isometry transform.
    auto to_transform() const -> Transform<T, D, TransformMode::Isometry> {
        Transform<T, D, TransformMode::Isometry> result;  // identity
        for (index_type r = 0; r < D; ++r) {
            for (index_type c = 0; c < D; ++c) {
                result(r, c) = m_mat(r, c);
            }
        }
        return result;
    }

    /// @brief Convert to an owning (D+1)x(D+1) matrix.
    auto to_matrix() const -> zipper::Matrix<T, D + 1, D + 1> {
        return to_transform().to_matrix();
    }

  private:
    zipper::Matrix<T, D, D> m_mat;
};

// Register Rotation as a Transform type.
namespace detail {
template <typename T, index_type D>
struct IsTransform<Rotation<T, D>> : std::true_type {};
} // namespace detail

// ============================================================================
// Rotation composition operators
// ============================================================================

/// @brief Rotation * Rotation → Rotation (matrix product).
template <typename T, index_type D>
auto operator*(const Rotation<T, D>& lhs, const Rotation<T, D>& rhs) {
    Rotation<T, D> result;
    for (index_type r = 0; r < D; ++r) {
        for (index_type c = 0; c < D; ++c) {
            T sum = T(0);
            for (index_type k = 0; k < D; ++k) {
                sum += lhs(r, k) * rhs(k, c);
            }
            result(r, c) = sum;
        }
    }
    return result;
}

/// @brief Rotation * Vector<D> → rotated point.
template <typename T, index_type D, zipper::concepts::Vector V>
    requires(std::decay_t<V>::extents_type::static_extent(0) == D)
auto operator*(const Rotation<T, D>& lhs, const V& rhs) {
    zipper::Vector<T, D> result;
    for (index_type r = 0; r < D; ++r) {
        T sum = T(0);
        for (index_type k = 0; k < D; ++k) {
            sum += lhs(r, k) * rhs(k);
        }
        result(r) = sum;
    }
    return result;
}

} // namespace zipper::transform
#endif
