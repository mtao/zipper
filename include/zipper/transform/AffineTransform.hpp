/// @file AffineTransform.hpp
/// @brief Owning affine transform type (4x4 matrix with [0 0 0 1] last row).
/// @ingroup transform
///
/// `AffineTransform<T>` is the primary user-facing affine transform type.
/// It owns its data via an `MDArray` storage backend and inherits the full
/// `AffineTransformBase` interface (affine-specific accessors, composition,
/// and all `MatrixBase` operations).
///
/// @code
///   using namespace zipper::transform;
///   AffineTransform<float> xform;  // default = identity
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
/// Default-constructs to the identity matrix.
template <typename T>
class AffineTransform
    : public detail::AffineTransformBase<
          zipper::expression::nullary::MDArray<
              T, zipper::extents<4, 4>,
              zipper::storage::matrix_layout<true>,
              zipper::default_accessor_policy<T>>> {
  public:
    using expression_type = zipper::expression::nullary::MDArray<
        T, zipper::extents<4, 4>,
        zipper::storage::matrix_layout<true>,
        zipper::default_accessor_policy<T>>;
    using Base = detail::AffineTransformBase<expression_type>;
    using value_type = T;
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

    /// @brief Construct from another AffineTransform.
    template <typename U>
    AffineTransform(const AffineTransform<U>& other) : Base() {
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

// Register AffineTransform as satisfying IsAffineTransform.
namespace detail {
template <typename T>
struct IsAffineTransform<AffineTransform<T>> : std::true_type {};
} // namespace detail

} // namespace zipper::transform
#endif
