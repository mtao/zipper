/// @file Scaling.hpp
/// @brief Specialized scaling transform with minimal storage.
/// @ingroup transform
///
/// `Scaling<T, D>` stores only a D-dimensional vector of scale factors,
/// rather than a full (D+1)x(D+1) matrix.  It satisfies `concepts::AffineTransform`
/// and can be composed with other transform types via `operator*`.
///
/// @code
///   using namespace zipper::transform;
///   Scaling<float, 3> s({2, 3, 4});
///   Vector<float, 3> p({1, 1, 1});
///   Vector<float, 3> q = s * p;  // q = {2, 3, 4}
///   auto inv = s.inverse();      // Scaling<float, 3>({0.5, 1/3, 0.25})
///   auto s2 = s * s;             // Scaling<float, 3>({4, 9, 16})
/// @endcode

#if !defined(ZIPPER_TRANSFORM_SCALING_HPP)
#define ZIPPER_TRANSFORM_SCALING_HPP

#include "detail/TransformBase.hpp"
#include <zipper/expression/unary/DiagonalEmbed.hpp>

namespace zipper::transform {

/// @brief Scaling transform with minimal storage (Vector<T, D>).
template <typename T, index_type D = 3>
class Scaling {
  public:
    using value_type = T;
    static constexpr index_type dim = D;
    static constexpr TransformMode mode = TransformMode::Affine;
    static constexpr index_type H = D + 1;
    using factors_expression_type = typename zipper::Vector<T, D>::expression_type;

    /// @brief Default-construct to uniform scale of 1 (identity).
    Scaling() : m_factors() {
        for (index_type i = 0; i < D; ++i) {
            m_factors(i) = T(1);
        }
    }

    /// @brief Construct from a vector of scale factors.
    template <zipper::concepts::Vector V>
        requires(std::decay_t<V>::extents_type::static_extent(0) == D)
    Scaling(const V& v) : m_factors() {
        for (index_type i = 0; i < D; ++i) {
            m_factors(i) = static_cast<T>(v(i));
        }
    }

    /// @brief Construct from an initializer list.
    Scaling(std::initializer_list<T> init) : m_factors(init) {}

    /// @brief Construct uniform scaling from a single scalar.
    explicit Scaling(T uniform) : m_factors() {
        for (index_type i = 0; i < D; ++i) {
            m_factors(i) = uniform;
        }
    }

    /// @brief Access the scale factors vector (const).
    auto const& factors() const { return m_factors; }
    /// @brief Access the scale factors vector (mutable).
    auto& factors() { return m_factors; }

    /// @brief Return a zero translation vector (concept API).
    ///
    /// Scaling has no translation component.
    auto translation() const { return zipper::Vector<T, D>{}; }

    /// @brief Return the diagonal matrix of scale factors (concept API).
    ///
    /// Returns a lazy DiagonalEmbed expression wrapping the factors vector.
    auto linear(this auto&& self) {
        using Self = decltype(self);
        using expr_t = typename std::decay_t<Self>::factors_expression_type;
        using child_t = zipper::detail::member_child_storage_t<Self, expr_t>;
        using diag_t = zipper::expression::unary::DiagonalEmbed<child_t>;
        return zipper::MatrixBase<diag_t>(
            std::in_place,
            std::forward<Self>(self).m_factors.expression());
    }

    /// @brief Access individual scale factor.
    T operator()(index_type i) const { return m_factors(i); }
    T& operator()(index_type i) { return m_factors(i); }

    /// @brief Inverse: reciprocal of each scale factor.
    Scaling inverse() const {
        Scaling result;
        for (index_type i = 0; i < D; ++i) {
            result.m_factors(i) = T(1) / m_factors(i);
        }
        return result;
    }

    /// @brief Convert to a full (D+1)x(D+1) AffineTransform.
    auto to_transform() const -> Transform<T, D, TransformMode::Affine> {
        Transform<T, D, TransformMode::Affine> result;  // identity
        for (index_type i = 0; i < D; ++i) {
            result(i, i) = m_factors(i);
        }
        return result;
    }

    /// @brief Convert to an owning (D+1)x(D+1) matrix.
    auto to_matrix() const -> zipper::Matrix<T, D + 1, D + 1> {
        return to_transform().to_matrix();
    }

  private:
    zipper::Vector<T, D> m_factors;
};

// Register Scaling as a Transform type.
namespace detail {
template <typename T, index_type D>
struct IsTransform<Scaling<T, D>> : std::true_type {};
} // namespace detail

// ============================================================================
// Scaling composition operators
// ============================================================================

/// @brief Scaling * Scaling → Scaling (component-wise product).
template <typename T, index_type D>
auto operator*(const Scaling<T, D>& lhs, const Scaling<T, D>& rhs) {
    Scaling<T, D> result;
    for (index_type i = 0; i < D; ++i) {
        result(i) = lhs(i) * rhs(i);
    }
    return result;
}

/// @brief Scaling * Vector<D> → scaled point.
template <typename T, index_type D, zipper::concepts::Vector V>
    requires(std::decay_t<V>::extents_type::static_extent(0) == D)
auto operator*(const Scaling<T, D>& lhs, const V& rhs) {
    zipper::Vector<T, D> result;
    for (index_type i = 0; i < D; ++i) {
        result(i) = lhs(i) * rhs(i);
    }
    return result;
}

} // namespace zipper::transform
#endif
