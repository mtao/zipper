/// @file QuaternionBase.hpp
/// @brief CRTP base for quaternion types with expression template support.
/// @ingroup user_types
///
/// `QuaternionBase<Expr>` provides the quaternion interface on top of a
/// rank-1, extent-4 expression.  Storage layout is (w, x, y, z) — scalar
/// first, matching the mathematical convention q = w + xi + yj + zk.
///
/// This follows the same CRTP architecture as VectorBase and FormBase:
///   - QuaternionBase wraps any qualifying expression.
///   - Quaternion<T> is the owning concrete type backed by MDArray.
///   - Concept `concepts::Quaternion` identifies quaternion wrappers.
///
/// Operations:
///   - Hamilton product via operator* (lazy, returns QuaternionBase<HamiltonProduct<...>>)
///   - conjugate(), inverse(), norm(), normalized(), normalize()
///   - dot() for quaternion inner product
///   - Component access: w(), x(), y(), z()
///
/// @see zipper::Quaternion — owning quaternion type.
/// @see zipper::concepts::Quaternion — concept for quaternion detection.
/// @see zipper::expression::binary::HamiltonProduct — lazy Hamilton product.

#if !defined(ZIPPER_QUATERNIONBASE_HPP)
#define ZIPPER_QUATERNIONBASE_HPP

#include "ZipperBase.hpp"
#include "as.hpp"
#include "concepts/Quaternion.hpp"
#include "concepts/detail/IsZipperBase.hpp"
#include "detail/assert.hpp"
#include "detail/member_child_storage.hpp"
#include "expression/binary/HamiltonProduct.hpp"
#include "expression/nullary/StlMDArray.hpp"
#include "expression/reductions/LpNorm.hpp"

#include <cmath>

namespace zipper {
template <typename ValueType> class Quaternion;

template <concepts::Expression Expr>
  requires(concepts::QualifiedRankedExpression<Expr, 1>)
class QuaternionBase : public ZipperBase<QuaternionBase, Expr> {
public:
  QuaternionBase() = default;

  using Base = ZipperBase<QuaternionBase, Expr>;
  using expression_type = typename Base::expression_type;
  using expression_traits = typename Base::expression_traits;
  using value_type = typename expression_traits::value_type;
  using extents_type = typename expression_traits::extents_type;
  using extents_traits = detail::ExtentsTraits<extents_type>;

  using Base::Base;
  using Base::cast;
  using Base::extent;
  using Base::extents;
  using Base::expression;

  auto eval() const { return Quaternion<value_type>(*this); }

  template <concepts::Quaternion Other>
  QuaternionBase(const Other &other) : QuaternionBase(other.expression()) {}

  QuaternionBase(concepts::QualifiedExpression auto &v)
    requires(expression::concepts::OwningExpression<expression_type>)
      : Base(v) {}
  QuaternionBase(concepts::QualifiedExpression auto &&v)
    requires(expression::concepts::OwningExpression<expression_type>)
      : Base(std::move(v)) {}
  QuaternionBase(const extents_type &e) : Base(e) {}

  auto operator=(concepts::QualifiedExpression auto const &v) -> QuaternionBase & {
    return Base::operator=(v);
  }

  template <typename... Args>
    requires(!(concepts::Quaternion<Args> && ...))
  QuaternionBase(Args &&...args)
      : Base(std::in_place, std::forward<Args>(args)...) {}

  template <concepts::Quaternion Other>
  auto operator=(const Other &other) -> QuaternionBase &
    requires(expression::concepts::WritableExpression<expression_type>)
  {
    expression().assign(other.expression());
    return *this;
  }

  [[nodiscard]] constexpr auto size() const -> index_type { return extent(0); }

  // ── Component access ──────────────────────────────────────────────────

  /// @brief Scalar (real) component.
  auto w() const -> value_type { return (*this)(0); }
  /// @brief First imaginary component (i).
  auto x() const -> value_type { return (*this)(1); }
  /// @brief Second imaginary component (j).
  auto y() const -> value_type { return (*this)(2); }
  /// @brief Third imaginary component (k).
  auto z() const -> value_type { return (*this)(3); }

  auto w() -> decltype(auto)
    requires(expression::concepts::WritableExpression<expression_type>)
  { return (*this)(0); }
  auto x() -> decltype(auto)
    requires(expression::concepts::WritableExpression<expression_type>)
  { return (*this)(1); }
  auto y() -> decltype(auto)
    requires(expression::concepts::WritableExpression<expression_type>)
  { return (*this)(2); }
  auto z() -> decltype(auto)
    requires(expression::concepts::WritableExpression<expression_type>)
  { return (*this)(3); }

  // ── Quaternion operations ─────────────────────────────────────────────

  /// @brief Quaternion conjugate: q* = (w, -x, -y, -z).
  auto conjugate() const -> Quaternion<value_type> {
    Quaternion<value_type> result;
    result.w() = w();
    result.x() = -x();
    result.y() = -y();
    result.z() = -z();
    return result;
  }

  /// @brief Quaternion norm (L2 norm of the 4-vector).
  auto norm() const -> value_type {
    return std::sqrt(w() * w() + x() * x() + y() * y() + z() * z());
  }

  /// @brief Squared norm (avoids the sqrt).
  auto squaredNorm() const -> value_type {
    return w() * w() + x() * x() + y() * y() + z() * z();
  }

  /// @brief Return a unit quaternion in the same direction.
  auto normalized() const -> Quaternion<value_type> {
    value_type n = norm();
    Quaternion<value_type> result;
    result.w() = w() / n;
    result.x() = x() / n;
    result.y() = y() / n;
    result.z() = z() / n;
    return result;
  }

  /// @brief Normalize in-place.
  void normalize()
    requires(expression::concepts::WritableExpression<expression_type>)
  {
    value_type n = norm();
    w() = w() / n;
    x() = x() / n;
    y() = y() / n;
    z() = z() / n;
  }

  /// @brief Quaternion multiplicative inverse: q^{-1} = q* / |q|^2.
  auto inverse() const -> Quaternion<value_type> {
    value_type n2 = squaredNorm();
    Quaternion<value_type> c = conjugate();
    Quaternion<value_type> result;
    result.w() = c.w() / n2;
    result.x() = c.x() / n2;
    result.y() = c.y() / n2;
    result.z() = c.z() / n2;
    return result;
  }

  /// @brief Quaternion dot product (4D inner product).
  auto dot(concepts::Quaternion auto const &o) const -> value_type {
    return w() * o.w() + x() * o.x() + y() * o.y() + z() * o.z();
  }

  // ── Semantic conversions ──────────────────────────────────────────────

  auto as_array() const { return zipper::as_array(*this); }
  auto as_tensor() const { return zipper::as_tensor(*this); }
  auto as_vector() const { return zipper::as_vector(*this); }

  // ── Iteration ─────────────────────────────────────────────────────────

  using Base::operator=;
};

// ── Hamilton product (lazy) ───────────────────────────────────────────────
//
// q1 * q2 returns a QuaternionBase<HamiltonProduct<...>> — the computation
// is deferred until coefficients are accessed, following the same pattern as
// VectorBase::cross() which returns VectorBase<CrossProduct<...>>.
//
// The child storage type is deduced from the forwarding reference: lvalue
// wrappers store by const reference, rvalue wrappers store by value.

namespace detail {
/// Deduce child expression storage for a forwarded Zipper wrapper.
/// Lvalue or reference-expression → const ExprType &; rvalue value-expression → ExprType.
template <typename Wrapper>
using quat_child_storage_t = std::conditional_t<
    std::is_lvalue_reference_v<Wrapper> ||
    std::is_reference_v<typename std::decay_t<Wrapper>::raw_expression_type>,
    const typename std::decay_t<Wrapper>::expression_type &,
    typename std::decay_t<Wrapper>::expression_type>;
} // namespace detail

template <typename ExprType1, typename ExprType2>
  requires(concepts::Quaternion<ExprType1> && concepts::Quaternion<ExprType2>)
auto operator*(ExprType1&& lhs, ExprType2&& rhs) {
    using lhs_child_t = detail::quat_child_storage_t<ExprType1>;
    using rhs_child_t = detail::quat_child_storage_t<ExprType2>;
    using HP = expression::binary::HamiltonProduct<lhs_child_t, rhs_child_t>;
    return QuaternionBase<HP>(std::in_place,
                              std::forward<ExprType1>(lhs).expression(),
                              std::forward<ExprType2>(rhs).expression());
}

// ── Scalar multiplication ─────────────────────────────────────────────────

template <concepts::Quaternion Q>
auto operator*(const Q &q, typename Q::value_type s) {
  using T = typename Q::value_type;
  Quaternion<T> result;
  result.w() = q.w() * s;
  result.x() = q.x() * s;
  result.y() = q.y() * s;
  result.z() = q.z() * s;
  return result;
}

template <concepts::Quaternion Q>
auto operator*(typename Q::value_type s, const Q &q) {
  return q * s;
}

// ── Scalar division ───────────────────────────────────────────────────────

template <concepts::Quaternion Q>
auto operator/(const Q &q, typename Q::value_type s) {
  using T = typename Q::value_type;
  Quaternion<T> result;
  result.w() = q.w() / s;
  result.x() = q.x() / s;
  result.y() = q.y() / s;
  result.z() = q.z() / s;
  return result;
}

// ── Addition ──────────────────────────────────────────────────────────────

template <concepts::Quaternion Q1, concepts::Quaternion Q2>
auto operator+(const Q1 &lhs, const Q2 &rhs) {
  using T = typename Q1::value_type;
  Quaternion<T> result;
  result.w() = lhs.w() + rhs.w();
  result.x() = lhs.x() + rhs.x();
  result.y() = lhs.y() + rhs.y();
  result.z() = lhs.z() + rhs.z();
  return result;
}

// ── Subtraction ───────────────────────────────────────────────────────────

template <concepts::Quaternion Q1, concepts::Quaternion Q2>
auto operator-(const Q1 &lhs, const Q2 &rhs) {
  using T = typename Q1::value_type;
  Quaternion<T> result;
  result.w() = lhs.w() - rhs.w();
  result.x() = lhs.x() - rhs.x();
  result.y() = lhs.y() - rhs.y();
  result.z() = lhs.z() - rhs.z();
  return result;
}

// ── Negation ──────────────────────────────────────────────────────────────

template <concepts::Quaternion Q>
auto operator-(const Q &q) {
  using T = typename Q::value_type;
  Quaternion<T> result;
  result.w() = -q.w();
  result.x() = -q.x();
  result.y() = -q.y();
  result.z() = -q.z();
  return result;
}

// ── Equality ──────────────────────────────────────────────────────────────

template <concepts::Quaternion Q1, concepts::Quaternion Q2>
auto operator==(const Q1 &lhs, const Q2 &rhs) -> bool {
  return lhs.w() == rhs.w() && lhs.x() == rhs.x() &&
         lhs.y() == rhs.y() && lhs.z() == rhs.z();
}

template <concepts::Quaternion Q1, concepts::Quaternion Q2>
auto operator!=(const Q1 &lhs, const Q2 &rhs) -> bool {
  return !(lhs == rhs);
}

// ── Deduction guides ──────────────────────────────────────────────────────

template <concepts::Expression Expr>
QuaternionBase(Expr &&) -> QuaternionBase<Expr>;
template <concepts::Expression Expr>
QuaternionBase(const Expr &) -> QuaternionBase<Expr>;

namespace concepts::detail {

template <typename T>
struct IsQuaternion<QuaternionBase<T>> : std::true_type {};
template <typename T>
struct IsZipperBase<QuaternionBase<T>> : std::true_type {};
} // namespace concepts::detail

} // namespace zipper
#endif
