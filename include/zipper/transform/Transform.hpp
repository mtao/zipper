/// @file Transform.hpp
/// @brief Owning transform type ((D+1)x(D+1) matrix with mode-aware semantics).
/// @ingroup transform
///
/// `Transform<T, D, Mode>` is the primary user-facing matrix-backed transform
/// type.  It owns its data via an `MDArray` storage backend and inherits
/// `MatrixBase` for all standard matrix operations.
///
/// Transform-specific accessors:
///   - `linear()`      — mutable/const view of the upper-left DxD block
///   - `translation()` — mutable/const view of the upper-right Dx1 column
///   - `matrix()`      — the full (D+1)x(D+1) matrix (as MatrixBase)
///   - `inverse()`     — mode-aware inverse
///
/// Template parameters:
///   - `T`: scalar type (e.g. `float`, `double`).
///   - `D`: spatial dimension (default 3).  The backing matrix is
///     `(D+1) x (D+1)`.
///   - `Mode`: `TransformMode` tag controlling inverse and vector-multiply
///     semantics.  Defaults to `TransformMode::Projective` (most general).
///   - `LayoutPolicy`: storage layout; defaults to column-major (OpenGL
///     convention).  Pass `zipper::storage::row_major` for row-major.
///   - `AccessorPolicy`: accessor policy; defaults to
///     `zipper::default_accessor_policy<T>`.
///
/// Convenience aliases:
///   - `AffineTransform<T, D>` = `Transform<T, D, TransformMode::Affine>`
///   - `Isometry<T, D>`        = `Transform<T, D, TransformMode::Isometry>`
///   - `ProjectiveTransform<T, D>` = `Transform<T, D, TransformMode::Projective>`
///
/// @code
///   using namespace zipper::transform;
///   AffineTransform<float> xform;       // 3D (4x4), col-major, Affine mode
///   Isometry<float> rigid;              // 3D (4x4), col-major, Isometry mode
///   Transform<float> proj;              // 3D (4x4), col-major, Projective mode
///   auto lin = xform.linear();          // DxD view
///   auto t   = xform.translation();     // D-vector view
///   auto inv = xform.inverse();         // mode-aware inverse
///   auto composed = xform * rigid;      // → AffineTransform (mode promotion)
/// @endcode

#if !defined(ZIPPER_TRANSFORM_TRANSFORM_HPP)
#define ZIPPER_TRANSFORM_TRANSFORM_HPP

#include "detail/TransformBase.hpp"
#include <zipper/expression/nullary/MDArray.hpp>
#include <zipper/storage/layout_types.hpp>

namespace zipper::transform {

/// @brief Owning transform backed by a (D+1)x(D+1) MDArray.
///
/// Default-constructs to the identity matrix.  Storage is column-major
/// by default, matching the OpenGL convention.
template <typename T, index_type D, TransformMode Mode,
          typename LayoutPolicy, typename AccessorPolicy>
class Transform
    : public zipper::MatrixBase<
          zipper::expression::nullary::MDArray<
              T, zipper::extents<D + 1, D + 1>,
              LayoutPolicy,
              AccessorPolicy>> {
  public:
    using expression_type = zipper::expression::nullary::MDArray<
        T, zipper::extents<D + 1, D + 1>,
        LayoutPolicy,
        AccessorPolicy>;
    using Base = zipper::MatrixBase<expression_type>;
    using value_type = T;
    using layout_type = LayoutPolicy;
    using accessor_type = AccessorPolicy;
    using extents_type = typename Base::extents_type;

    /// @brief The spatial dimension (the matrix is (D+1)x(D+1)).
    static constexpr index_type dim = D;
    /// @brief Homogeneous size = D + 1.
    static constexpr index_type H = D + 1;
    /// @brief The transform mode.
    static constexpr TransformMode mode = Mode;

    using Base::expression;
    using Base::extent;
    using Base::extents;

    /// @brief Default-construct to identity.
    Transform() : Base() {
        for (zipper::index_type i = 0; i <= D; ++i) {
            (*this)(i, i) = T(1);
        }
    }

    /// @brief Construct from any matrix expression (copies element-by-element).
    template <zipper::concepts::Matrix M>
        requires(M::extents_type::static_extent(0) == D + 1 &&
                 M::extents_type::static_extent(1) == D + 1)
    Transform(const M& other) : Base() {
        for (zipper::index_type r = 0; r <= D; ++r) {
            for (zipper::index_type c = 0; c <= D; ++c) {
                (*this)(r, c) = static_cast<T>(other(r, c));
            }
        }
    }

    /// @brief Construct from another Transform (possibly different
    ///        scalar type, mode, layout, or accessor policy).
    template <typename U, TransformMode M2, typename LP, typename AP>
    Transform(const Transform<U, D, M2, LP, AP>& other) : Base() {
        for (zipper::index_type r = 0; r <= D; ++r) {
            for (zipper::index_type c = 0; c <= D; ++c) {
                (*this)(r, c) = static_cast<T>(other(r, c));
            }
        }
    }

    /// @brief Copy constructor.
    Transform(const Transform& other) : Base() {
        for (zipper::index_type r = 0; r <= D; ++r) {
            for (zipper::index_type c = 0; c <= D; ++c) {
                (*this)(r, c) = other(r, c);
            }
        }
    }

    /// @brief Copy assignment.
    auto operator=(const Transform& other) -> Transform& {
        for (zipper::index_type r = 0; r <= D; ++r) {
            for (zipper::index_type c = 0; c <= D; ++c) {
                (*this)(r, c) = other(r, c);
            }
        }
        return *this;
    }

    /// @brief Assign from another Transform (possibly different
    ///        scalar type, mode, layout, or accessor policy).
    template <typename U, TransformMode M2, typename LP, typename AP>
    auto operator=(const Transform<U, D, M2, LP, AP>& other) -> Transform& {
        for (zipper::index_type r = 0; r <= D; ++r) {
            for (zipper::index_type c = 0; c <= D; ++c) {
                (*this)(r, c) = static_cast<T>(other(r, c));
            }
        }
        return *this;
    }

    /// @brief Assign from any matrix expression.
    template <zipper::concepts::Matrix M>
        requires(M::extents_type::static_extent(0) == D + 1 &&
                 M::extents_type::static_extent(1) == D + 1)
    auto operator=(const M& other) -> Transform& {
        for (zipper::index_type r = 0; r <= D; ++r) {
            for (zipper::index_type c = 0; c <= D; ++c) {
                (*this)(r, c) = static_cast<T>(other(r, c));
            }
        }
        return *this;
    }

    // ========================================================================
    // Transform-specific accessors
    // ========================================================================

    /// @brief Evaluate to a copy of this transform.
    auto eval() const { return Transform(*this); }

    /// @brief Access the full (D+1)x(D+1) matrix as a MatrixBase.
    auto matrix() const -> const Base& { return *this; }
    auto matrix() -> Base& { return *this; }

    /// @brief View of the upper-left DxD (rotation + scale) block.
    auto linear(this auto&& self) {
        return std::forward<decltype(self)>(self)
            .template slice<zipper::static_slice_t<0, D>,
                            zipper::static_slice_t<0, D>>();
    }

    /// @brief View of the translation column (top D entries of column D).
    auto translation(this auto&& self) {
        return std::forward<decltype(self)>(self)
            .slice(zipper::static_slice_t<0, D>{},
                   zipper::static_index_t<D>{});
    }

    /// @brief Convert to an owning Matrix<T, D+1, D+1>.
    auto to_matrix() const -> zipper::Matrix<T, D + 1, D + 1> {
        zipper::Matrix<T, D + 1, D + 1> result;
        for (zipper::index_type r = 0; r <= D; ++r) {
            for (zipper::index_type c = 0; c <= D; ++c) {
                result(r, c) = (*this)(r, c);
            }
        }
        return result;
    }

    // ========================================================================
    // Inverse
    // ========================================================================

    /// @brief Mode-aware inverse.
    ///
    /// Dispatches based on the TransformMode:
    ///   - Projective: full (D+1)x(D+1) matrix inverse
    ///   - Affine: exploits affine structure [L^-1, -L^-1*t; 0 1]
    ///   - Isometry: exploits orthogonality [R^T, -R^T*t; 0 1]
    auto inverse() const -> Transform {
        if constexpr (Mode == TransformMode::Isometry) {
            return rotation_inverse_();
        } else if constexpr (Mode == TransformMode::Affine) {
            return affine_inverse_();
        } else {
            return projective_inverse_();
        }
    }

  private:
    /// @brief Full (D+1)x(D+1) matrix inverse.
    auto projective_inverse_() const -> Transform {
        zipper::Matrix<T, H, H> mat;
        for (zipper::index_type r = 0; r < H; ++r) {
            for (zipper::index_type c = 0; c < H; ++c) {
                mat(r, c) = (*this)(r, c);
            }
        }
        zipper::Matrix<T, H, H> inv_mat = zipper::utils::inverse(mat);

        Transform result;
        for (zipper::index_type r = 0; r < H; ++r) {
            for (zipper::index_type c = 0; c < H; ++c) {
                result(r, c) = inv_mat(r, c);
            }
        }
        return result;
    }

    /// @brief Affine inverse: [L^-1  -L^-1*t; 0 1].
    auto affine_inverse_() const -> Transform {
        zipper::Matrix<T, D, D> lin;
        for (zipper::index_type r = 0; r < D; ++r) {
            for (zipper::index_type c = 0; c < D; ++c) {
                lin(r, c) = (*this)(r, c);
            }
        }
        zipper::Matrix<T, D, D> lin_inv = zipper::utils::inverse(lin);

        zipper::Vector<T, D> t;
        for (zipper::index_type i = 0; i < D; ++i) {
            t(i) = (*this)(i, D);
        }

        Transform result;
        for (zipper::index_type r = 0; r < D; ++r) {
            for (zipper::index_type c = 0; c < D; ++c) {
                result(r, c) = lin_inv(r, c);
            }
            T sum = T(0);
            for (zipper::index_type k = 0; k < D; ++k) {
                sum += lin_inv(r, k) * t(k);
            }
            result(r, D) = -sum;
        }
        for (zipper::index_type c = 0; c < D; ++c) {
            result(D, c) = T(0);
        }
        result(D, D) = T(1);

        return result;
    }

    /// @brief Isometry inverse: [R^T  -R^T*t; 0 1].
    auto rotation_inverse_() const -> Transform {
        zipper::Vector<T, D> t;
        for (zipper::index_type i = 0; i < D; ++i) {
            t(i) = (*this)(i, D);
        }

        Transform result;
        for (zipper::index_type r = 0; r < D; ++r) {
            for (zipper::index_type c = 0; c < D; ++c) {
                result(r, c) = (*this)(c, r);
            }
            T sum = T(0);
            for (zipper::index_type k = 0; k < D; ++k) {
                sum += result(r, k) * t(k);
            }
            result(r, D) = -sum;
        }
        for (zipper::index_type c = 0; c < D; ++c) {
            result(D, c) = T(0);
        }
        result(D, D) = T(1);

        return result;
    }
};

// Register all Transform instantiations as satisfying IsTransform and IsMatrixTransform.
namespace detail {
template <typename T, index_type D, TransformMode Mode, typename LP, typename AP>
struct IsTransform<Transform<T, D, Mode, LP, AP>> : std::true_type {};

template <typename T, index_type D, TransformMode Mode, typename LP, typename AP>
struct IsMatrixTransform<Transform<T, D, Mode, LP, AP>> : std::true_type {};

/// @brief Specialize TransformTraits for owning Transform so that
///        methods like inverse() preserve the layout/accessor policies.
template <typename T, index_type D, TransformMode Mode, typename LP, typename AP>
struct TransformTraits<Transform<T, D, Mode, LP, AP>> {
    using value_type = T;
    static constexpr index_type dim = D;
    static constexpr TransformMode mode = Mode;
    using owning_type = Transform<T, D, Mode, LP, AP>;
};

} // namespace detail

// ============================================================================
// Convenience aliases
// ============================================================================

/// @brief Alias for affine transforms (last row = [0 ... 0 1], general
///        linear block).
template <typename T,
          index_type D = 3,
          typename LayoutPolicy = zipper::storage::matrix_layout<false>,
          typename AccessorPolicy = zipper::default_accessor_policy<T>>
using AffineTransform = Transform<T, D, TransformMode::Affine, LayoutPolicy, AccessorPolicy>;

/// @brief Alias for isometry transforms (last row = [0 ... 0 1],
///        orthogonal linear block — pure rotation + translation).
template <typename T,
          index_type D = 3,
          typename LayoutPolicy = zipper::storage::matrix_layout<false>,
          typename AccessorPolicy = zipper::default_accessor_policy<T>>
using Isometry = Transform<T, D, TransformMode::Isometry, LayoutPolicy, AccessorPolicy>;

/// @brief Alias for projective transforms (most general — no structural
///        assumptions about the matrix).
template <typename T,
          index_type D = 3,
          typename LayoutPolicy = zipper::storage::matrix_layout<false>,
          typename AccessorPolicy = zipper::default_accessor_policy<T>>
using ProjectiveTransform = Transform<T, D, TransformMode::Projective, LayoutPolicy, AccessorPolicy>;

} // namespace zipper::transform
#endif
