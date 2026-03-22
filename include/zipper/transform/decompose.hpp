/// @file decompose.hpp
/// @brief Transform decomposition: extract Translation, Rotation, Scaling.
/// @ingroup transform
///
/// Decomposes an affine transform into its constituent Translation,
/// Rotation, and Scaling components.  This is the inverse of composing
/// `T * R * S` via the factories in model.hpp.
///
/// The decomposition assumes the linear block contains no shear — only
/// rotation and axis-aligned scale.  If the linear block has negative
/// determinant (a reflection), one scale factor will be negative.
///
/// @code
///   using namespace zipper::transform;
///   auto T = translation(Vector<float,3>({1, 2, 3}));
///   auto R = rotation(radians(45.0f), Vector<float,3>({0, 1, 0}));
///   auto S = scaling(Vector<float,3>({2, 3, 4}));
///   AffineTransform<float> xf = T * R * S;
///
///   auto [t, r, s] = trs_decompose(xf);
///   // t ≈ Translation({1, 2, 3})
///   // r ≈ Rotation(45° about Y)
///   // s ≈ Scaling({2, 3, 4})
/// @endcode
///
/// For 3D transforms, convert the Rotation to a Quaternion:
/// @code
///   auto q = to_quaternion(r.matrix());  // from quaternion_transform.hpp
/// @endcode
///
/// @see zipper::transform::Translation
/// @see zipper::transform::Rotation
/// @see zipper::transform::Scaling
/// @see zipper::transform::model.hpp — composition factories.

#if !defined(ZIPPER_TRANSFORM_DECOMPOSE_HPP)
#define ZIPPER_TRANSFORM_DECOMPOSE_HPP

#include <cmath>
#include <tuple>

#include "Translation.hpp"
#include "Rotation.hpp"
#include "Scaling.hpp"
#include "detail/TransformBase.hpp"

namespace zipper::transform {

/// @brief Decompose an affine transform into Translation, Rotation, Scaling.
///
/// Given an affine transform whose linear block is `R * S` (rotation times
/// diagonal scale, no shear), extracts the three components:
///
///   1. **Translation**: the last column of the transform.
///   2. **Scale**: column norms of the linear block (one per axis).
///      If the determinant of the linear block is negative, the first
///      scale factor is negated to absorb the reflection.
///   3. **Rotation**: the linear block with scale divided out.
///
/// @tparam A  An AffineTransform (Affine or Isometry mode).
/// @param  xf The transform to decompose.
/// @return    `std::tuple<Translation<T,D>, Rotation<T,D>, Scaling<T,D>>`.
template<concepts::AffineTransform A>
auto trs_decompose(const A &xf)
  -> std::tuple<
    Translation<typename std::decay_t<A>::value_type, std::decay_t<A>::dim>,
    Rotation<typename std::decay_t<A>::value_type, std::decay_t<A>::dim>,
    Scaling<typename std::decay_t<A>::value_type, std::decay_t<A>::dim>> {

    using Decayed = std::decay_t<A>;
    using T = typename Decayed::value_type;
    constexpr index_type D = Decayed::dim;

    // ── 1. Extract translation ──────────────────────────────────────
    Translation<T, D> t;
    {
        auto tv = xf.translation();
        for (index_type i = 0; i < D; ++i) {
            t(i) = static_cast<T>(tv(i));
        }
    }

    // ── 2. Extract column norms as scale factors ────────────────────
    auto lin = xf.linear();

    Vector<T, D> scale_factors;
    for (index_type c = 0; c < D; ++c) {
        T sum = T(0);
        for (index_type r = 0; r < D; ++r) {
            T v = static_cast<T>(lin(r, c));
            sum += v * v;
        }
        scale_factors(c) = std::sqrt(sum);
    }

    // Handle reflection: if the determinant of the linear block is
    // negative, negate the first scale factor so the rotation matrix
    // has positive determinant (proper rotation).
    // Compute determinant via cofactor expansion along the first row.
    // This is dimension-agnostic for D=2 and D=3; for general D we
    // would need a more general determinant, but D>3 transforms are
    // uncommon in practice.
    if constexpr (D == 2) {
        T det = static_cast<T>(lin(0, 0)) * static_cast<T>(lin(1, 1))
                - static_cast<T>(lin(0, 1)) * static_cast<T>(lin(1, 0));
        if (det < T(0)) {
            scale_factors(0) = -scale_factors(0);
        }
    } else if constexpr (D == 3) {
        T det = static_cast<T>(lin(0, 0)) * (static_cast<T>(lin(1, 1)) * static_cast<T>(lin(2, 2)) - static_cast<T>(lin(1, 2)) * static_cast<T>(lin(2, 1)))
                - static_cast<T>(lin(0, 1)) * (static_cast<T>(lin(1, 0)) * static_cast<T>(lin(2, 2)) - static_cast<T>(lin(1, 2)) * static_cast<T>(lin(2, 0)))
                + static_cast<T>(lin(0, 2)) * (static_cast<T>(lin(1, 0)) * static_cast<T>(lin(2, 1)) - static_cast<T>(lin(1, 1)) * static_cast<T>(lin(2, 0)));
        if (det < T(0)) {
            scale_factors(0) = -scale_factors(0);
        }
    }
    // For D > 3, we skip the determinant check (no reflection handling).

    // ── 3. Build pure rotation: divide each column by its scale ─────
    Rotation<T, D> r;
    for (index_type c = 0; c < D; ++c) {
        T s = scale_factors(c);
        T inv_s = (std::abs(s) > std::numeric_limits<T>::epsilon() * T(100))
                    ? (T(1) / s)
                    : T(0);
        for (index_type row = 0; row < D; ++row) {
            r(row, c) = static_cast<T>(lin(row, c)) * inv_s;
        }
    }

    Scaling<T, D> sc;
    for (index_type i = 0; i < D; ++i) {
        sc(i) = scale_factors(i);
    }

    return { t, r, sc };
}

}// namespace zipper::transform
#endif
