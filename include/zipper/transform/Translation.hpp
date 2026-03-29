/// @file Translation.hpp
/// @brief Specialized translation transform with minimal storage.
/// @ingroup transform
///
/// `Translation<T, D>` stores only a D-dimensional translation vector,
/// rather than a full (D+1)x(D+1) matrix.  It satisfies `concepts::Isometry`
/// and can be composed with other transform types via `operator*`.
///
/// @code
///   using namespace zipper::transform;
///   Translation<float, 3> t({1, 2, 3});
///   Vector<float, 3> p({0, 0, 0});
///   Vector<float, 3> q = t * p;  // q = {1, 2, 3}
///   auto inv = t.inverse();      // Translation<float, 3>({-1, -2, -3})
///   auto t2 = t * t;             // Translation<float, 3>({2, 4, 6})
/// @endcode

#if !defined(ZIPPER_TRANSFORM_TRANSLATION_HPP)
#define ZIPPER_TRANSFORM_TRANSLATION_HPP

#include "detail/TransformBase.hpp"
#include <zipper/expression/nullary/Identity.hpp>

namespace zipper::transform {

/// @brief Translation transform with minimal storage (Vector<T, D>).
template <typename T, index_type D = 3>
class Translation {
  public:
    using value_type = T;
    static constexpr index_type dim = D;
    static constexpr TransformMode mode = TransformMode::Isometry;
    // Provide D and H for compatibility with generic code
    static constexpr index_type H = D + 1;

    /// @brief Default-construct to zero translation (identity).
    Translation() : m_vec() {}

    /// @brief Construct from a vector.
    template <zipper::concepts::Vector V>
        requires(std::decay_t<V>::extents_type::static_extent(0) == D)
    Translation(const V& v) : m_vec() {
        for (index_type i = 0; i < D; ++i) {
            m_vec(i) = static_cast<T>(v(i));
        }
    }

    /// @brief Construct from an initializer list.
    Translation(std::initializer_list<T> init) : m_vec(init) {}

    /// @brief Access the translation vector (const).
    auto const& vector() const { return m_vec; }
    /// @brief Access the translation vector (mutable).
    auto& vector() { return m_vec; }

    /// @brief Return the translation vector (concept API).
    auto const& translation() const { return m_vec; }
    /// @brief Return the translation vector (mutable, concept API).
    auto& translation() { return m_vec; }

    /// @brief Return the identity matrix as the linear part (concept API).
    ///
    /// Translation has no linear component, so this returns a DxD identity.
    auto linear() const {
        return zipper::MatrixBase<zipper::expression::nullary::Identity<T, D, D>>{};
    }

    /// @brief Access individual components.
    T operator()(index_type i) const { return m_vec(i); }
    T& operator()(index_type i) { return m_vec(i); }

    /// @brief Inverse: negate the translation.
    Translation inverse() const {
        Translation result;
        for (index_type i = 0; i < D; ++i) {
            result.m_vec(i) = -m_vec(i);
        }
        return result;
    }

    /// @brief Convert to a full (D+1)x(D+1) Isometry transform.
    auto to_transform() const -> Transform<T, D, TransformMode::Isometry> {
        Transform<T, D, TransformMode::Isometry> result;  // identity
        for (index_type i = 0; i < D; ++i) {
            result(i, D) = m_vec(i);
        }
        return result;
    }

    /// @brief Convert to an owning (D+1)x(D+1) matrix.
    auto to_matrix() const -> zipper::Matrix<T, D + 1, D + 1> {
        return to_transform().to_matrix();
    }

  private:
    zipper::Vector<T, D> m_vec;
};

// Register Translation as a Transform type.
namespace detail {
template <typename T, index_type D>
struct IsTransform<Translation<T, D>> : std::true_type {};
} // namespace detail

// ============================================================================
// Translation composition operators
// ============================================================================

/// @brief Translation * Translation → Translation (sum).
template <typename T, index_type D>
auto operator*(const Translation<T, D>& lhs, const Translation<T, D>& rhs) {
    Translation<T, D> result;
    for (index_type i = 0; i < D; ++i) {
        result(i) = lhs(i) + rhs(i);
    }
    return result;
}

/// @brief Translation * Vector<D> → translated point.
template <typename T, index_type D, zipper::concepts::Vector V>
    requires(std::decay_t<V>::extents_type::static_extent(0) == D)
auto operator*(const Translation<T, D>& lhs, const V& rhs) {
    zipper::Vector<T, D> result;
    for (index_type i = 0; i < D; ++i) {
        result(i) = lhs(i) + rhs(i);
    }
    return result;
}

} // namespace zipper::transform
#endif
