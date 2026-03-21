/// @file TransformBase.hpp
/// @brief CRTP base for transform types ((D+1)x(D+1) matrix with mode-aware
///        semantics).
/// @ingroup transform
///
/// `TransformBase<Expr, Mode>` inherits from `MatrixBase<Expr>` to get all
/// standard matrix operations (arithmetic, slicing, etc.) and adds
/// transform-specific accessors and operations:
///
///   - `linear()`      — mutable/const view of the upper-left DxD block
///   - `translation()` — mutable/const view of the upper-right Dx1 column
///   - `matrix()`      — the full (D+1)x(D+1) matrix (as MatrixBase)
///   - `inverse()`     — mode-aware inverse:
///       * Projective: full (D+1)x(D+1) matrix inverse
///       * Affine:     [L^-1, -L^-1*t; 0 1]
///       * Isometry:   [R^T, -R^T*t; 0 1]
///   - `affine_inverse()` — efficient inverse exploiting affine structure
///   - `rotation_inverse()` — fast inverse for rigid-body transforms
///
/// The spatial dimension D is deduced from the expression's static extents
/// as `static_extent(0) - 1`.
///
/// The `TransformMode` parameter controls:
///   - Which inverse algorithm `inverse()` dispatches to
///   - How `operator*` with a D-dimensional vector behaves
///   - Mode promotion rules during composition
///
/// Composition (`operator*`) between two Transform types is eager and
/// returns an owning `Transform<T, D, promoted_mode>`.

#if !defined(ZIPPER_TRANSFORM_DETAIL_TRANSFORMBASE_HPP)
#define ZIPPER_TRANSFORM_DETAIL_TRANSFORMBASE_HPP

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/MatrixBase.hxx>
#include <zipper/storage/layout_types.hpp>
#include <zipper/utils/inverse.hpp>

namespace zipper::transform {

/// @brief Mode tag for Transform types.
///
/// Controls inverse computation strategy and vector multiplication semantics.
///   - `Projective`: most general — no assumptions about the last row.
///   - `Affine`: last row is [0 ... 0 1], linear block may contain
///     scale/shear.
///   - `Isometry`: last row is [0 ... 0 1], linear block is orthogonal
///     (pure rotation).
enum class TransformMode {
    Projective,
    Affine,
    Isometry
};

/// @brief Return the least restrictive of two TransformModes.
///
/// Promotion rules (Projective > Affine > Isometry):
///   - Isometry * Isometry → Isometry
///   - Isometry * Affine   → Affine
///   - Affine   * Affine   → Affine
///   - anything * Projective → Projective
constexpr TransformMode promote_mode(TransformMode a, TransformMode b) {
    // Projective=0, Affine=1, Isometry=2 — lower numeric value = less
    // restrictive.  We want the minimum (least restrictive).
    using U = std::underlying_type_t<TransformMode>;
    return static_cast<TransformMode>(
        std::min(static_cast<U>(a), static_cast<U>(b)));
}

// Forward declaration of owning type.
template <typename T,
          index_type D = 3,
          TransformMode Mode = TransformMode::Projective,
          typename LayoutPolicy = zipper::storage::matrix_layout<false>,
          typename AccessorPolicy = zipper::default_accessor_policy<T>>
class Transform;

namespace detail {

/// @brief Trait to identify Transform types.
template <typename> struct IsTransform : std::false_type {};

/// @brief Trait to extract the owning Transform type from a concrete
///        Transform or TransformBase.  Defaults to
///        `Transform<T, D, Mode>` (default layout/accessor) when the
///        expression does not carry policy information.
template <typename Derived>
struct TransformTraits {
    using value_type = typename std::decay_t<Derived>::value_type;
    static constexpr index_type dim = std::decay_t<Derived>::extents_type::static_extent(0) - 1;
    static constexpr TransformMode mode = std::decay_t<Derived>::mode;
    using owning_type = transform::Transform<value_type, dim, mode>;
};

} // namespace detail

namespace concepts {

/// @brief Concept matching any Transform wrapper (all modes).
template <typename T>
concept Transform = detail::IsTransform<std::decay_t<T>>::value;

/// @brief Concept matching Affine or Isometry transforms (not Projective).
template <typename T>
concept AffineTransform = Transform<T> &&
    (std::decay_t<T>::mode == TransformMode::Affine ||
     std::decay_t<T>::mode == TransformMode::Isometry);

/// @brief Concept matching only Isometry transforms.
template <typename T>
concept Isometry = Transform<T> &&
    std::decay_t<T>::mode == TransformMode::Isometry;

} // namespace concepts

namespace detail {

/// @brief CRTP base providing transform-specific operations on top of MatrixBase.
///
/// The underlying expression must be rank-2 with static extents (D+1)x(D+1)
/// for some D >= 1.  D is deduced as `static_extent(0) - 1`.
template <zipper::concepts::Expression Expr, TransformMode Mode>
    requires(zipper::concepts::QualifiedRankedExpression<Expr, 2> &&
             std::decay_t<Expr>::extents_type::static_extent(0) != zipper::dynamic_extent &&
             std::decay_t<Expr>::extents_type::static_extent(0) ==
             std::decay_t<Expr>::extents_type::static_extent(1) &&
             std::decay_t<Expr>::extents_type::static_extent(0) >= 2)
class TransformBase : public zipper::MatrixBase<Expr> {
  public:
    using Base = zipper::MatrixBase<Expr>;
    using expression_type = typename Base::expression_type;
    using expression_traits = typename Base::expression_traits;
    using value_type = typename expression_traits::value_type;
    using extents_type = typename expression_traits::extents_type;

    /// @brief The spatial dimension D (the matrix is (D+1)x(D+1)).
    static constexpr index_type D = extents_type::static_extent(0) - 1;
    /// @brief Homogeneous size = D + 1.
    static constexpr index_type H = D + 1;
    /// @brief The transform mode.
    static constexpr TransformMode mode = Mode;

    TransformBase() = default;

    using Base::Base;
    using Base::expression;
    using Base::extent;
    using Base::extents;

    /// @brief Construct from any expression (forwarded to MatrixBase).
    template <typename... Args>
        requires(!(concepts::Transform<Args> && ...))
    TransformBase(Args&&... args)
        : Base(std::forward<Args>(args)...) {}

    /// @brief Construct from another Transform.
    template <concepts::Transform Other>
    TransformBase(const Other& other)
        : Base(other.expression()) {}

    auto operator=(concepts::Transform auto const& v) -> TransformBase& {
        Base::operator=(v.expression());
        return *this;
    }

    template <zipper::concepts::Expression Other>
    auto operator=(const Other& other) -> TransformBase&
        requires(zipper::expression::concepts::WritableExpression<expression_type>)
    {
        Base::operator=(other);
        return *this;
    }

    /// @brief Evaluate to an owning Transform.
    auto eval() const { return transform::Transform<value_type, D, Mode>(*this); }

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

    /// @brief Mode-aware inverse.
    ///
    /// Dispatches based on the TransformMode:
    ///   - Projective: full (D+1)x(D+1) matrix inverse
    ///   - Affine: exploits affine structure [L^-1, -L^-1*t; 0 1]
    ///   - Isometry: exploits orthogonality [R^T, -R^T*t; 0 1]
    auto inverse(this auto const& self) {
        if constexpr (Mode == TransformMode::Isometry) {
            return self.rotation_inverse();
        } else if constexpr (Mode == TransformMode::Affine) {
            return self.affine_inverse();
        } else {
            return self.projective_inverse();
        }
    }

    /// @brief Full (D+1)x(D+1) matrix inverse.
    ///
    /// Uses the general inverse (QR-based or closed-form for small sizes).
    /// This is the most expensive inverse but works for any transform,
    /// including projective transforms where the last row is not [0...0 1].
    auto projective_inverse(this auto const& self) {
        using Derived = std::decay_t<decltype(self)>;
        using T = typename Derived::value_type;
        // Compute the full matrix inverse via zipper::utils::inverse
        zipper::Matrix<T, H, H> mat;
        for (zipper::index_type r = 0; r < H; ++r) {
            for (zipper::index_type c = 0; c < H; ++c) {
                mat(r, c) = self(r, c);
            }
        }
        zipper::Matrix<T, H, H> inv_mat = zipper::utils::inverse(mat);

        // Build the result Transform with the same mode
        using result_type = typename TransformTraits<Derived>::owning_type;
        result_type result;
        for (zipper::index_type r = 0; r < H; ++r) {
            for (zipper::index_type c = 0; c < H; ++c) {
                result(r, c) = inv_mat(r, c);
            }
        }
        return result;
    }

    /// @brief Compute the affine inverse efficiently.
    ///
    /// For an affine matrix `[L t; 0 1]`, the inverse is
    /// `[L^-1  -L^-1*t; 0 1]`.  Uses a full DxD inverse of the linear
    /// block, so it is correct even when the transform includes
    /// non-uniform scaling.
    ///
    /// The returned Transform preserves the layout and accessor
    /// policies of the derived type when available.
    ///
    /// @see rotation_inverse — faster path when the linear block is
    ///      known to be orthogonal (pure rotation, no scale/shear).
    auto affine_inverse(this auto const& self) {
        using Derived = std::decay_t<decltype(self)>;
        using result_type = typename TransformTraits<Derived>::owning_type;
        using T = typename Derived::value_type;

        // Extract the DxD linear part and invert it
        zipper::Matrix<T, D, D> lin;
        for (zipper::index_type r = 0; r < D; ++r) {
            for (zipper::index_type c = 0; c < D; ++c) {
                lin(r, c) = self(r, c);
            }
        }
        zipper::Matrix<T, D, D> lin_inv = zipper::utils::inverse(lin);

        // Compute -L^-1 * t
        zipper::Vector<T, D> t;
        for (zipper::index_type i = 0; i < D; ++i) {
            t(i) = self(i, D);
        }

        result_type result;
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
        // Last row: [0 ... 0 1]
        for (zipper::index_type c = 0; c < D; ++c) {
            result(D, c) = T(0);
        }
        result(D, D) = T(1);

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
    ///
    /// The returned Transform preserves the layout and accessor
    /// policies of the derived type when available.
    auto rotation_inverse(this auto const& self) {
        using Derived = std::decay_t<decltype(self)>;
        using result_type = typename TransformTraits<Derived>::owning_type;
        using T = typename Derived::value_type;

        zipper::Vector<T, D> t;
        for (zipper::index_type i = 0; i < D; ++i) {
            t(i) = self(i, D);
        }

        result_type result;
        // R^T: swap rows and columns of the upper-left DxD
        for (zipper::index_type r = 0; r < D; ++r) {
            for (zipper::index_type c = 0; c < D; ++c) {
                result(r, c) = self(c, r);
            }
            // -R^T * t
            T sum = T(0);
            for (zipper::index_type k = 0; k < D; ++k) {
                sum += result(r, k) * t(k);
            }
            result(r, D) = -sum;
        }
        // Last row: [0 ... 0 1]
        for (zipper::index_type c = 0; c < D; ++c) {
            result(D, c) = T(0);
        }
        result(D, D) = T(1);

        return result;
    }
};

// Register TransformBase as satisfying IsTransform.
template <typename T, TransformMode M>
struct IsTransform<TransformBase<T, M>> : std::true_type {};

} // namespace detail

// ============================================================================
// Operator overloads
// ============================================================================

/// @brief Eager transform composition: A * B → Transform with promoted mode.
///
/// For Affine/Isometry modes, exploits affine structure:
///   [R1 t1; 0 1] * [R2 t2; 0 1] = [R1*R2  R1*t2+t1; 0 1]
///
/// For Projective mode (either operand), performs full matrix multiply.
template <concepts::Transform A, concepts::Transform B>
    requires(std::decay_t<A>::D == std::decay_t<B>::D)
auto operator*(const A& lhs, const B& rhs) {
    using TA = std::decay_t<A>;
    using TB = std::decay_t<B>;
    using T = typename TA::value_type;

    constexpr index_type D = TA::D;
    constexpr index_type H = D + 1;
    constexpr TransformMode ResultMode = promote_mode(TA::mode, TB::mode);

    using result_type = Transform<T, D, ResultMode>;
    result_type result;

    if constexpr (ResultMode == TransformMode::Projective) {
        // Full (D+1)x(D+1) matrix multiply — no structural assumptions.
        for (zipper::index_type r = 0; r < H; ++r) {
            for (zipper::index_type c = 0; c < H; ++c) {
                T sum = T(0);
                for (zipper::index_type k = 0; k < H; ++k) {
                    sum += lhs(r, k) * rhs(k, c);
                }
                result(r, c) = sum;
            }
        }
    } else {
        // Affine/Isometry: exploit [R t; 0 1] structure.
        // R = R1 * R2  (DxD block)
        for (zipper::index_type r = 0; r < D; ++r) {
            for (zipper::index_type c = 0; c < D; ++c) {
                T sum = T(0);
                for (zipper::index_type k = 0; k < D; ++k) {
                    sum += lhs(r, k) * rhs(k, c);
                }
                result(r, c) = sum;
            }
            // t = R1 * t2 + t1
            T sum = T(0);
            for (zipper::index_type k = 0; k < D; ++k) {
                sum += lhs(r, k) * rhs(k, D);
            }
            result(r, D) = sum + lhs(r, D);
        }
        // Last row: [0 ... 0 1]
        for (zipper::index_type c = 0; c < D; ++c) {
            result(D, c) = T(0);
        }
        result(D, D) = T(1);
    }

    return result;
}

/// @brief Transform (Affine or Isometry) * Vector (D-dimensional) — affine action.
///
/// Applies the transform to a D-dimensional vector as a point:
///   result = L * v + t
/// where L is the DxD linear block and t is the translation column.
/// Returns a Vector<T, D>.
template <concepts::Transform A, zipper::concepts::Vector V>
    requires(std::decay_t<A>::mode != TransformMode::Projective &&
             std::decay_t<V>::extents_type::static_extent(0) == std::decay_t<A>::D)
auto operator*(const A& lhs, const V& rhs) {
    using T = typename std::decay_t<A>::value_type;
    constexpr index_type D = std::decay_t<A>::D;

    zipper::Vector<T, D> result;
    for (zipper::index_type r = 0; r < D; ++r) {
        T sum = lhs(r, D); // translation component
        for (zipper::index_type k = 0; k < D; ++k) {
            sum += lhs(r, k) * rhs(k);
        }
        result(r) = sum;
    }
    return result;
}

/// @brief Transform (Projective) * Vector (D-dimensional) — projective action.
///
/// Extends the D-vector to homogeneous coordinates (w=1), performs full
/// (D+1)x(D+1) matrix-vector multiply, then divides by the resulting w
/// component (perspective division).  Returns a Vector<T, D>.
template <concepts::Transform A, zipper::concepts::Vector V>
    requires(std::decay_t<A>::mode == TransformMode::Projective &&
             std::decay_t<V>::extents_type::static_extent(0) == std::decay_t<A>::D)
auto operator*(const A& lhs, const V& rhs) {
    using T = typename std::decay_t<A>::value_type;
    constexpr index_type D = std::decay_t<A>::D;
    constexpr index_type H = D + 1;

    // Full (D+1)-dim multiply with implicit w=1
    zipper::Vector<T, H> tmp;
    for (zipper::index_type r = 0; r < H; ++r) {
        T sum = lhs(r, D); // contribution from w=1
        for (zipper::index_type k = 0; k < D; ++k) {
            sum += lhs(r, k) * rhs(k);
        }
        tmp(r) = sum;
    }

    // Perspective division
    T w = tmp(D);
    zipper::Vector<T, D> result;
    for (zipper::index_type i = 0; i < D; ++i) {
        result(i) = tmp(i) / w;
    }
    return result;
}

/// @brief Transform * Vector (homogeneous, D+1) — matrix-vector
///        multiply in homogeneous coordinates.
///
/// Delegates to the standard (D+1)x(D+1) matrix-vector product via
/// MatrixBase.  No perspective division is performed; the caller is
/// responsible for interpreting the result.
template <concepts::Transform A, zipper::concepts::Vector V>
    requires(std::decay_t<V>::extents_type::static_extent(0) == std::decay_t<A>::H)
auto operator*(const A& lhs, const V& rhs) {
    return static_cast<const typename std::decay_t<A>::Base&>(lhs) * rhs;
}

} // namespace zipper::transform
#endif
