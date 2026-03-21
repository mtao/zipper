/// @file TransformBase.hpp
/// @brief Shared infrastructure for the transform system: mode enum, traits,
///        concepts, and matrix-backed transform composition operators.
/// @ingroup transform
///
/// This header provides:
///   - `TransformMode` enum and `promote_mode()` function
///   - `IsTransform` / `IsMatrixTransform` type traits
///   - `concepts::Transform`, `concepts::MatrixTransform`,
///     `concepts::AffineTransform`, `concepts::Isometry`
///   - `operator*` overloads for composing and applying matrix-backed transforms

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

/// @brief Trait to identify Transform types (any transform, including specialized).
template <typename> struct IsTransform : std::false_type {};

/// @brief Trait to identify matrix-backed Transform types (Transform).
///
/// Types with full (D+1)x(D+1) matrix storage satisfy this; specialized types
/// like Translation, Scaling, Rotation do not.
template <typename> struct IsMatrixTransform : std::false_type {};

/// @brief Trait to extract the owning Transform type from a concrete Transform.
template <typename Derived>
struct TransformTraits {
    using value_type = typename std::decay_t<Derived>::value_type;
    static constexpr index_type dim = std::decay_t<Derived>::dim;
    static constexpr TransformMode mode = std::decay_t<Derived>::mode;
    using owning_type = transform::Transform<value_type, dim, mode>;
};

} // namespace detail

namespace concepts {

/// @brief Concept matching any Transform wrapper (all modes, including specialized).
template <typename T>
concept Transform = detail::IsTransform<std::decay_t<T>>::value;

/// @brief Concept matching matrix-backed transforms (full (D+1)x(D+1) storage).
///
/// These are `Transform` types that support full `(r,c)` indexing.
/// Specialized types (Translation, Scaling, Rotation, AxisAngleRotation) do NOT
/// satisfy this concept.
template <typename T>
concept MatrixTransform = detail::IsMatrixTransform<std::decay_t<T>>::value;

/// @brief Concept matching Affine or Isometry transforms (not Projective).
///
/// Requires structural API: `.linear()` returning a matrix-like type and
/// `.translation()` returning a vector-like type, in addition to mode checking.
template <typename T>
concept AffineTransform = Transform<T> &&
    (std::decay_t<T>::mode == TransformMode::Affine ||
     std::decay_t<T>::mode == TransformMode::Isometry) &&
    requires(const std::decay_t<T>& t) {
        { t.linear() };
        { t.translation() };
    };

/// @brief Concept matching only Isometry transforms.
///
/// Requires structural API: `.linear()` and `.translation()`, plus
/// Isometry mode tag.
template <typename T>
concept Isometry = Transform<T> &&
    std::decay_t<T>::mode == TransformMode::Isometry &&
    requires(const std::decay_t<T>& t) {
        { t.linear() };
        { t.translation() };
    };

} // namespace concepts

// ============================================================================
// Operator overloads for matrix-backed transforms
// ============================================================================

/// @brief Eager transform composition: A * B → Transform with promoted mode.
///
/// For Affine/Isometry modes, exploits affine structure:
///   [R1 t1; 0 1] * [R2 t2; 0 1] = [R1*R2  R1*t2+t1; 0 1]
///
/// For Projective mode (either operand), performs full matrix multiply.
template <concepts::MatrixTransform A, concepts::MatrixTransform B>
    requires(std::decay_t<A>::dim == std::decay_t<B>::dim)
auto operator*(const A& lhs, const B& rhs) {
    using TA = std::decay_t<A>;
    using TB = std::decay_t<B>;
    using T = typename TA::value_type;

    constexpr index_type D = TA::dim;
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
template <concepts::MatrixTransform A, zipper::concepts::Vector V>
    requires(std::decay_t<A>::mode != TransformMode::Projective &&
             std::decay_t<V>::extents_type::static_extent(0) == std::decay_t<A>::dim)
auto operator*(const A& lhs, const V& rhs) {
    using T = typename std::decay_t<A>::value_type;
    constexpr index_type D = std::decay_t<A>::dim;

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
template <concepts::MatrixTransform A, zipper::concepts::Vector V>
    requires(std::decay_t<A>::mode == TransformMode::Projective &&
             std::decay_t<V>::extents_type::static_extent(0) == std::decay_t<A>::dim)
auto operator*(const A& lhs, const V& rhs) {
    using T = typename std::decay_t<A>::value_type;
    constexpr index_type D = std::decay_t<A>::dim;
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
template <concepts::MatrixTransform A, zipper::concepts::Vector V>
    requires(std::decay_t<V>::extents_type::static_extent(0) == std::decay_t<A>::H)
auto operator*(const A& lhs, const V& rhs) {
    return static_cast<const typename std::decay_t<A>::Base&>(lhs) * rhs;
}

} // namespace zipper::transform
#endif
