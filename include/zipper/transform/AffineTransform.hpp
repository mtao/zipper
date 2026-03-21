/// @file AffineTransform.hpp
/// @brief Owning affine transform type (4x4 matrix with [0 0 0 1] last row).
/// @ingroup transform
///
/// `AffineTransform<T>` is the primary user-facing affine transform type.
/// It owns its data via an `MDArray` storage backend and inherits the full
/// `AffineTransformBase` interface (affine-specific accessors, composition,
/// and all `MatrixBase` operations).
///
/// Template parameters:
///   - `T`: scalar type (e.g. `float`, `double`).
///   - `LayoutPolicy`: storage layout; defaults to column-major (OpenGL
///     convention).  Pass `zipper::storage::row_major` for row-major.
///   - `AccessorPolicy`: accessor policy; defaults to
///     `zipper::default_accessor_policy<T>`.
///
/// @code
///   using namespace zipper::transform;
///   AffineTransform<float> xform;  // default = identity, col-major
///   auto lin = xform.linear();     // 3x3 view
///   auto t = xform.translation();  // 3-vector view
///   auto inv = xform.affine_inverse();
///   auto composed = xform * xform; // affine composition
/// @endcode

#if !defined(ZIPPER_TRANSFORM_AFFINETRANSFORM_HPP)
#define ZIPPER_TRANSFORM_AFFINETRANSFORM_HPP

#include "detail/AffineTransformBase.hpp"
#include <zipper/expression/nullary/MDArray.hpp>
#include <zipper/storage/layout_types.hpp>

namespace zipper::transform {

/// @brief Owning affine transform backed by a 4x4 MDArray.
///
/// Default-constructs to the identity matrix.  Storage is column-major
/// by default, matching the OpenGL convention.
template <typename T, typename LayoutPolicy, typename AccessorPolicy>
class AffineTransform
    : public detail::AffineTransformBase<
          zipper::expression::nullary::MDArray<
              T, zipper::extents<4, 4>,
              LayoutPolicy,
              AccessorPolicy>> {
  public:
    using expression_type = zipper::expression::nullary::MDArray<
        T, zipper::extents<4, 4>,
        LayoutPolicy,
        AccessorPolicy>;
    using Base = detail::AffineTransformBase<expression_type>;
    using value_type = T;
    using layout_type = LayoutPolicy;
    using accessor_type = AccessorPolicy;
    using extents_type = typename Base::extents_type;

    using Base::expression;
    using Base::extent;
    using Base::extents;

    /// @brief Default-construct to identity.
    AffineTransform() : Base() {
        (*this)(0, 0) = T(1);
        (*this)(1, 1) = T(1);
        (*this)(2, 2) = T(1);
        (*this)(3, 3) = T(1);
    }

    /// @brief Construct from any matrix expression (copies element-by-element).
    template <zipper::concepts::Matrix M>
        requires(M::extents_type::static_extent(0) == 4 &&
                 M::extents_type::static_extent(1) == 4)
    AffineTransform(const M& other) : Base() {
        for (zipper::index_type r = 0; r < 4; ++r) {
            for (zipper::index_type c = 0; c < 4; ++c) {
                (*this)(r, c) = static_cast<T>(other(r, c));
            }
        }
    }

    /// @brief Construct from another AffineTransform (possibly different
    ///        scalar type, layout, or accessor policy).
    template <typename U, typename LP, typename AP>
    AffineTransform(const AffineTransform<U, LP, AP>& other) : Base() {
        for (zipper::index_type r = 0; r < 4; ++r) {
            for (zipper::index_type c = 0; c < 4; ++c) {
                (*this)(r, c) = static_cast<T>(other(r, c));
            }
        }
    }

    /// @brief Construct from an AffineTransformBase (expression).
    template <zipper::concepts::Expression Expr>
        requires(concepts::AffineTransform<detail::AffineTransformBase<Expr>>)
    AffineTransform(const detail::AffineTransformBase<Expr>& other) : Base() {
        for (zipper::index_type r = 0; r < 4; ++r) {
            for (zipper::index_type c = 0; c < 4; ++c) {
                (*this)(r, c) = static_cast<T>(other(r, c));
            }
        }
    }

    /// @brief Copy constructor.
    AffineTransform(const AffineTransform& other) : Base() {
        for (zipper::index_type r = 0; r < 4; ++r) {
            for (zipper::index_type c = 0; c < 4; ++c) {
                (*this)(r, c) = other(r, c);
            }
        }
    }

    /// @brief Copy assignment.
    auto operator=(const AffineTransform& other) -> AffineTransform& {
        for (zipper::index_type r = 0; r < 4; ++r) {
            for (zipper::index_type c = 0; c < 4; ++c) {
                (*this)(r, c) = other(r, c);
            }
        }
        return *this;
    }

    /// @brief Assign from another AffineTransform (possibly different
    ///        scalar type, layout, or accessor policy).
    template <typename U, typename LP, typename AP>
    auto operator=(const AffineTransform<U, LP, AP>& other) -> AffineTransform& {
        for (zipper::index_type r = 0; r < 4; ++r) {
            for (zipper::index_type c = 0; c < 4; ++c) {
                (*this)(r, c) = static_cast<T>(other(r, c));
            }
        }
        return *this;
    }

    /// @brief Assign from any matrix expression.
    template <zipper::concepts::Matrix M>
        requires(M::extents_type::static_extent(0) == 4 &&
                 M::extents_type::static_extent(1) == 4)
    auto operator=(const M& other) -> AffineTransform& {
        for (zipper::index_type r = 0; r < 4; ++r) {
            for (zipper::index_type c = 0; c < 4; ++c) {
                (*this)(r, c) = static_cast<T>(other(r, c));
            }
        }
        return *this;
    }

    /// @brief Convert to an owning Matrix<T, 4, 4>.
    auto to_matrix() const -> zipper::Matrix<T, 4, 4> {
        zipper::Matrix<T, 4, 4> result;
        for (zipper::index_type r = 0; r < 4; ++r) {
            for (zipper::index_type c = 0; c < 4; ++c) {
                result(r, c) = (*this)(r, c);
            }
        }
        return result;
    }
};

// Register all AffineTransform instantiations as satisfying IsAffineTransform.
namespace detail {
template <typename T, typename LP, typename AP>
struct IsAffineTransform<AffineTransform<T, LP, AP>> : std::true_type {};

/// @brief Specialize AffineTransformTraits for owning AffineTransform
///        so that methods like affine_inverse() preserve the
///        layout/accessor policies.
template <typename T, typename LP, typename AP>
struct AffineTransformTraits<AffineTransform<T, LP, AP>> {
    using value_type = T;
    using owning_type = AffineTransform<T, LP, AP>;
};

} // namespace detail

} // namespace zipper::transform
#endif
