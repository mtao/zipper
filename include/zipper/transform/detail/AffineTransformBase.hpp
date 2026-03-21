/// @file AffineTransformBase.hpp
/// @brief CRTP base for affine transform types (4x4 matrix with [0 0 0 1] last row).
/// @ingroup transform
///
/// `AffineTransformBase<Expr>` inherits from `MatrixBase<Expr>` to get all
/// standard matrix operations (arithmetic, slicing, etc.) and adds
/// affine-specific accessors and operations:
///
///   - `linear()`      — mutable/const view of the upper-left 3x3 block
///   - `translation()` — mutable/const view of the upper-right 3x1 column
///   - `matrix()`      — the full 4x4 matrix (as MatrixBase)
///   - `affine_inverse()` — efficient inverse exploiting affine structure
///
/// This type lives in `zipper::transform::detail` and is not registered as a
/// core zipper type (no IsZipperBase/IsMatrix specialization beyond what
/// MatrixBase already provides).
///
/// Composition (`operator*`) between two AffineTransform types is eager and
/// returns an owning `AffineTransform<T>`.

#if !defined(ZIPPER_TRANSFORM_DETAIL_AFFINETRANSFORMBASE_HPP)
#define ZIPPER_TRANSFORM_DETAIL_AFFINETRANSFORMBASE_HPP

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/MatrixBase.hxx>
#include <zipper/utils/inverse.hpp>

namespace zipper::transform {

// Forward declaration of owning type.
template <typename T> class AffineTransform;

namespace detail {

/// @brief Trait to identify AffineTransform types.
template <typename> struct IsAffineTransform : std::false_type {};

} // namespace detail

namespace concepts {

/// @brief Concept matching any AffineTransform wrapper.
template <typename T>
concept AffineTransform = detail::IsAffineTransform<std::decay_t<T>>::value;

} // namespace concepts

namespace detail {

/// @brief CRTP base providing affine-specific operations on top of MatrixBase.
///
/// The underlying expression must be rank-2 with static extents 4x4.
template <zipper::concepts::Expression Expr>
    requires(zipper::concepts::QualifiedRankedExpression<Expr, 2> &&
             std::decay_t<Expr>::extents_type::static_extent(0) == 4 &&
             std::decay_t<Expr>::extents_type::static_extent(1) == 4)
class AffineTransformBase : public zipper::MatrixBase<Expr> {
  public:
    using Base = zipper::MatrixBase<Expr>;
    using expression_type = typename Base::expression_type;
    using expression_traits = typename Base::expression_traits;
    using value_type = typename expression_traits::value_type;
    using extents_type = typename expression_traits::extents_type;

    AffineTransformBase() = default;

    using Base::Base;
    using Base::expression;
    using Base::extent;
    using Base::extents;

    /// @brief Construct from any expression (forwarded to MatrixBase).
    template <typename... Args>
        requires(!(concepts::AffineTransform<Args> && ...))
    AffineTransformBase(Args&&... args)
        : Base(std::forward<Args>(args)...) {}

    /// @brief Construct from another AffineTransform.
    template <concepts::AffineTransform Other>
    AffineTransformBase(const Other& other)
        : Base(other.expression()) {}

    auto operator=(concepts::AffineTransform auto const& v) -> AffineTransformBase& {
        Base::operator=(v.expression());
        return *this;
    }

    template <zipper::concepts::Expression Other>
    auto operator=(const Other& other) -> AffineTransformBase&
        requires(zipper::expression::concepts::WritableExpression<expression_type>)
    {
        Base::operator=(other);
        return *this;
    }

    /// @brief Evaluate to an owning AffineTransform.
    auto eval() const { return transform::AffineTransform<value_type>(*this); }

    /// @brief Access the full 4x4 matrix as a MatrixBase.
    auto matrix() const -> const Base& { return *this; }
    auto matrix() -> Base& { return *this; }

    /// @brief View of the upper-left 3x3 (rotation + scale) block.
    auto linear(this auto&& self) {
        return std::forward<decltype(self)>(self)
            .template slice<zipper::static_slice_t<0, 3>,
                            zipper::static_slice_t<0, 3>>();
    }

    /// @brief View of the translation column (top 3 entries of column 3).
    auto translation(this auto&& self) {
        return std::forward<decltype(self)>(self)
            .slice(zipper::static_slice_t<0, 3>{},
                   zipper::static_index_t<3>{});
    }

    /// @brief Compute the affine inverse efficiently.
    ///
    /// For an affine matrix `[L t; 0 1]`, the inverse is
    /// `[L^-1  -L^-1*t; 0 1]`.  Uses a full 3x3 inverse of the linear
    /// block, so it is correct even when the transform includes
    /// non-uniform scaling.
    ///
    /// @see rotation_inverse — faster path when the linear block is
    ///      known to be orthogonal (pure rotation, no scale/shear).
    auto affine_inverse() const -> transform::AffineTransform<value_type> {
        using T = value_type;

        // Extract the 3x3 linear part and invert it
        zipper::Matrix<T, 3, 3> lin;
        for (zipper::index_type r = 0; r < 3; ++r) {
            for (zipper::index_type c = 0; c < 3; ++c) {
                lin(r, c) = (*this)(r, c);
            }
        }
        zipper::Matrix<T, 3, 3> lin_inv = zipper::utils::inverse(lin);

        // Compute -L^-1 * t
        T tx = (*this)(0, 3);
        T ty = (*this)(1, 3);
        T tz = (*this)(2, 3);

        transform::AffineTransform<T> result;
        for (zipper::index_type r = 0; r < 3; ++r) {
            for (zipper::index_type c = 0; c < 3; ++c) {
                result(r, c) = lin_inv(r, c);
            }
            result(r, 3) = -(lin_inv(r, 0) * tx +
                             lin_inv(r, 1) * ty +
                             lin_inv(r, 2) * tz);
        }
        result(3, 0) = T(0);
        result(3, 1) = T(0);
        result(3, 2) = T(0);
        result(3, 3) = T(1);

        return result;
    }

    /// @brief Compute the inverse assuming the linear block is orthogonal.
    ///
    /// For a rigid-body transform `[R t; 0 1]` where `R` is an orthogonal
    /// rotation matrix, `R^-1 = R^T`.  The inverse is then
    /// `[R^T  -R^T*t; 0 1]`.
    ///
    /// This is significantly cheaper than `affine_inverse()` (no matrix
    /// inverse — just a transpose and a matrix-vector multiply), but
    /// produces incorrect results if the linear block contains
    /// non-uniform scale or shear.
    auto rotation_inverse() const -> transform::AffineTransform<value_type> {
        using T = value_type;

        T tx = (*this)(0, 3);
        T ty = (*this)(1, 3);
        T tz = (*this)(2, 3);

        transform::AffineTransform<T> result;
        // R^T: swap rows and columns of the upper-left 3x3
        for (zipper::index_type r = 0; r < 3; ++r) {
            for (zipper::index_type c = 0; c < 3; ++c) {
                result(r, c) = (*this)(c, r);
            }
            // -R^T * t
            result(r, 3) = -(result(r, 0) * tx +
                              result(r, 1) * ty +
                              result(r, 2) * tz);
        }
        result(3, 0) = T(0);
        result(3, 1) = T(0);
        result(3, 2) = T(0);
        result(3, 3) = T(1);

        return result;
    }
};

// Register AffineTransformBase as satisfying IsAffineTransform.
template <typename T>
struct IsAffineTransform<AffineTransformBase<T>> : std::true_type {};

} // namespace detail

/// @brief Eager affine composition: A * B → AffineTransform.
///
/// Exploits affine structure: for [R1 t1; 0 1] * [R2 t2; 0 1],
/// the result is [R1*R2  R1*t2+t1; 0 1].
template <concepts::AffineTransform A, concepts::AffineTransform B>
auto operator*(const A& lhs, const B& rhs)
    -> AffineTransform<typename A::value_type> {
    using T = typename A::value_type;

    AffineTransform<T> result;

    // R = R1 * R2  (3x3 block)
    for (zipper::index_type r = 0; r < 3; ++r) {
        for (zipper::index_type c = 0; c < 3; ++c) {
            result(r, c) = lhs(r, 0) * rhs(0, c) +
                           lhs(r, 1) * rhs(1, c) +
                           lhs(r, 2) * rhs(2, c);
        }
        // t = R1 * t2 + t1
        result(r, 3) = lhs(r, 0) * rhs(0, 3) +
                       lhs(r, 1) * rhs(1, 3) +
                       lhs(r, 2) * rhs(2, 3) +
                       lhs(r, 3);
    }
    result(3, 0) = T(0);
    result(3, 1) = T(0);
    result(3, 2) = T(0);
    result(3, 3) = T(1);

    return result;
}

/// @brief AffineTransform * Vector — apply the transform to a point/direction.
///
/// Delegates to the standard matrix-vector product via MatrixBase.
template <concepts::AffineTransform A, zipper::concepts::Vector V>
    requires(std::decay_t<V>::extents_type::static_extent(0) == 4)
auto operator*(const A& lhs, const V& rhs) {
    return static_cast<const typename std::decay_t<A>::Base&>(lhs) * rhs;
}

/// @brief AffineTransform * Matrix — e.g. view * projection.
///
/// Delegates to the standard matrix-matrix product via MatrixBase.
/// Returns a plain Matrix, not an AffineTransform, since the result
/// of affine * arbitrary-matrix is not necessarily affine.
template <concepts::AffineTransform A, zipper::concepts::Matrix M>
    requires(std::decay_t<M>::extents_type::static_extent(0) == 4 &&
             std::decay_t<M>::extents_type::static_extent(1) == 4)
auto operator*(const A& lhs, const M& rhs) {
    return static_cast<const typename std::decay_t<A>::Base&>(lhs) * rhs;
}

/// @brief Matrix * AffineTransform — e.g. projection * view.
///
/// Delegates to the standard matrix-matrix product via MatrixBase.
template <zipper::concepts::Matrix M, concepts::AffineTransform A>
    requires(std::decay_t<M>::extents_type::static_extent(0) == 4 &&
             std::decay_t<M>::extents_type::static_extent(1) == 4)
auto operator*(const M& lhs, const A& rhs) {
    return lhs * static_cast<const typename std::decay_t<A>::Base&>(rhs);
}

} // namespace zipper::transform
#endif
