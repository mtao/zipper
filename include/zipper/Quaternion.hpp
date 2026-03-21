#if !defined(ZIPPER_QUATERNION_HPP)
#define ZIPPER_QUATERNION_HPP

/// @file Quaternion.hpp
/// @brief Owning quaternion type with static extent-4 storage.
/// @ingroup user_types
///
/// `Quaternion<T>` is the primary user-facing quaternion type.  It owns its
/// data via an `MDArray<T, extents<4>>` storage backend and inherits the full
/// `QuaternionBase` interface (Hamilton product, conjugate, inverse, norm,
/// normalized, dot, component access w/x/y/z).
///
/// Storage layout is scalar-first: (w, x, y, z), matching the mathematical
/// convention q = w + xi + yj + zk.
///
/// Construction:
///   - Default: `Quaternion<double> q;` (zero-initialised).
///   - From components: `Quaternion<double> q(1.0, 0.0, 0.0, 0.0);` (identity).
///   - From initializer list: `Quaternion<double> q({1.0, 0.0, 0.0, 0.0});`
///   - Copy from expression: `Quaternion<double> q(some_expression);`
///
/// @code
///   Quaternion<float> q(1.0f, 0.0f, 0.0f, 0.0f);  // identity
///   Quaternion<float> r(0.0f, 1.0f, 0.0f, 0.0f);   // 180° around x
///   auto product = q * r;                            // Hamilton product
///   auto conj = q.conjugate();
///   auto inv = q.inverse();
///   float n = q.norm();
///   auto u = q.normalized();
/// @endcode
///
/// @see zipper::QuaternionBase — CRTP base providing the quaternion interface.
/// @see zipper::concepts::Quaternion — concept for quaternion detection.
/// @see zipper::expression::binary::HamiltonProduct — lazy Hamilton product.

#include "QuaternionBase.hpp"
#include "zipper/detail/assert.hpp"
#include "zipper/expression/nullary/MDArray.hpp"
#include "zipper/types.hpp"

namespace zipper {

template <typename ValueType>
class Quaternion
    : public QuaternionBase<expression::nullary::MDArray<ValueType, zipper::extents<4>>> {
public:
  using expression_type =
      expression::nullary::MDArray<ValueType, zipper::extents<4>>;
  using Base = QuaternionBase<expression_type>;
  using Base::expression;
  using value_type = Base::value_type;
  using extents_type = Base::extents_type;
  using Base::extent;
  using Base::extents;

  Quaternion() = default;
  Quaternion(const Quaternion &o) = default;
  Quaternion &operator=(const Quaternion &o) = default;
  Quaternion(Quaternion &&o) = default;
  Quaternion &operator=(Quaternion &&o) {
    expression().operator=(std::move(o.expression()));
    return *this;
  }

  /// @brief Construct from four scalar components (w, x, y, z).
  Quaternion(value_type w, value_type x, value_type y, value_type z) {
    (*this)(0) = w;
    (*this)(1) = x;
    (*this)(2) = y;
    (*this)(3) = z;
  }

  template <concepts::Quaternion Other>
  Quaternion(const Other &other) : Base(other) {}

  template <concepts::Expression Other>
  Quaternion(const Other &other) : Base(other) {}

  template <typename T>
  Quaternion(const std::initializer_list<T> &l) {
    ZIPPER_ASSERT(l.size() == 4);
    index_type i = 0;
    for (auto &v : l) {
      (*this)(i++) = static_cast<value_type>(v);
    }
  }

  using Base::operator=;
};

// ── Deduction guides ────────────────────────────────────────────────────

template <concepts::Quaternion QB>
Quaternion(const QB &o) -> Quaternion<std::decay_t<typename QB::value_type>>;

template <concepts::Expression Expr>
  requires(Expr::extents_type::rank() == 1)
Quaternion(const Expr &o) -> Quaternion<std::decay_t<typename Expr::value_type>>;

// ── Concept registrations ───────────────────────────────────────────────

namespace concepts::detail {
template <typename ValueType>
struct IsQuaternion<zipper::Quaternion<ValueType>> : std::true_type {};
template <typename ValueType>
struct IsZipperBase<zipper::Quaternion<ValueType>> : std::true_type {};
} // namespace concepts::detail

} // namespace zipper

#endif
