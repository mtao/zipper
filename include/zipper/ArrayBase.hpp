#if !defined(ZIPPER_ARRAYBASE_HPP)
#define ZIPPER_ARRAYBASE_HPP

#include "ZipperBase.hpp"
#include "concepts/detail/IsZipperBase.hpp"
#include "expression/reductions/All.hpp"
#include "expression/reductions/Any.hpp"
#include "expression/reductions/CoefficientProduct.hpp"
#include "expression/reductions/CoefficientSum.hpp"
#include "expression/reductions/LpNorm.hpp"
#include "expression/reductions/LpNormPowered.hpp"
#include "zipper/types.hpp"
//
#include "concepts/Array.hpp"
#include "concepts/Expression.hpp"
//
////
#include "detail/extents/static_extents_to_integral_sequence.hpp"
#include "expression/nullary/MDSpan.hpp"
#include "expression/unary/Abs.hpp"
#include "expression/unary/ScalarPower.hpp"
#include "expression/unary/detail/operation_implementations.hpp"
#include "zipper/detail/declare_operations.hpp"
#include "zipper/expression/binary/ArithmeticExpressions.hpp"
#include "zipper/expression/unary/ScalarArithmetic.hpp"

namespace zipper {

template <concepts::Expression Expr>
class ArrayBase : public ZipperBase<ArrayBase, Expr> {
public:
  ArrayBase() = default;

  using expression_type = std::decay_t<Expr>;
  using expression_traits =
      expression::detail::ExpressionTraits<expression_type>;
  using value_type = typename expression_traits::value_type;
  using extents_type = typename expression_traits::extents_type;
  using extents_traits = detail::ExtentsTraits<extents_type>;
  using Base = ZipperBase<ArrayBase, Expr>;
  using Base::Base;
  using Base::expression;

  template <index_type... N>
  auto eval(const std::integer_sequence<index_type, N...> &) const
    requires(std::is_same_v<extents<N...>, extents_type>)
  {
    return Array_<value_type, zipper::extents<N...>>(this->expression());
  }
  auto eval() const {
    return eval(
        detail::extents::static_extents_to_integral_sequence_t<extents_type>{});
  }
  auto operator=(concepts::Array auto const &v) -> ArrayBase & {
    return Base::operator=(v.expression());
  }
  auto operator=(concepts::Array auto &&v) -> ArrayBase & {
    return Base::operator=(v.expression());
  }

  template <concepts::Array Other>
  ArrayBase(const Other &other)
    requires(expression_traits::is_writable)
      : ArrayBase(other.expression()) {}

  auto operator*=(const value_type &other) -> ArrayBase &
    requires(expression_traits::is_writable)
  {
    return *this = other * *this;
  }
  auto operator/=(const value_type &other) -> ArrayBase &
    requires(expression_traits::is_writable)
  {
    return *this = *this / other;
  }
  template <concepts::Array Other>
  auto operator+=(const Other &other) -> ArrayBase &
    requires(expression_traits::is_writable)
  {
    return *this = *this + other;
  }
  template <concepts::Array Other>
  auto operator-=(const Other &other) -> ArrayBase &
    requires(expression_traits::is_writable)
  {
    return *this = *this - other;
  }
  template <concepts::Array Other>
  auto operator*=(const Other &other) -> ArrayBase &
    requires(expression_traits::is_writable)
  {
    return *this = *this * other;
  }
  template <concepts::Array Other>
  auto operator/=(const Other &other) -> ArrayBase &
    requires(expression_traits::is_writable)
  {
    return *this = *this / other;
  }

  //--------------------------------------------------------------
  //

  auto pow(value_type const &exp) const {
    return ArrayBase<
        expression::unary::ScalarPower<const expression_type, value_type>>(
        expression(), exp);
  }

  auto abs() const {
    return ArrayBase<expression::unary::Abs<const expression_type>>(
        expression());
  }

  auto sum() const -> value_type {
    return expression::reductions::CoefficientSum{expression()}();
  }

  auto product() const -> value_type {
    return expression::reductions::CoefficientProduct{expression()}();
  }

  template <index_type T> auto norm_powered() const -> value_type {
    return expression::reductions::LpNormPowered<T, const expression_type>(
        expression())();
  }
  auto norm_powered(value_type T) const -> value_type {
    return pow(T).abs().sum();
  }

  template <index_type T = 2> auto norm() const -> value_type {
    return expression::reductions::LpNorm<T, const expression_type>(
        expression())();
  }
  auto norm(value_type T) const -> value_type {
    value_type p = value_type(1.0) / T;
    return std::pow<value_type>(norm_powered(T), p);
  }

  template <index_type T = 2> auto normalized() const {
    return *this / norm<T>();
  }
  auto normalized(value_type T) const -> value_type { return *this / norm(T); }
  template <index_type T = 2> void normalize() { *this /= norm<T>(); }
  void normalize(value_type T) { *this /= norm(T); }

  [[nodiscard]] auto any() const -> bool
    requires(std::is_same_v<value_type, bool>)
  {
    return expression::reductions::Any(expression())();
  }
  [[nodiscard]] auto all() const -> bool
    requires(std::is_same_v<value_type, bool>)
  {
    return expression::reductions::All(expression())();
  }

  // Slice methods - delegate to ZipperBase::slice_expression
  template <typename... Slices> auto slice() {
    auto v = Base::template slice_expression<Slices...>();
    return ArrayBase<std::decay_t<decltype(v)>>(std::move(v));
  }
  template <typename... Slices> auto slice(Slices &&...slices) const {
    auto v =
        Base::template slice_expression<Slices...>(std::forward<Slices>(slices)...);
    return ArrayBase<std::decay_t<decltype(v)>>(std::move(v));
  }
  template <typename... Slices> auto slice() const {
    auto v = Base::template slice_expression<Slices...>();
    return ArrayBase<std::decay_t<decltype(v)>>(std::move(v));
  }
  template <typename... Slices> auto slice(Slices &&...slices) {
    auto v =
        Base::template slice_expression<Slices...>(std::forward<Slices>(slices)...);
    return ArrayBase<std::decay_t<decltype(v)>>(std::move(v));
  }
};

template <concepts::Expression Expr>
ArrayBase(Expr &&) -> ArrayBase<Expr>;
template <concepts::Expression Expr>
ArrayBase(const Expr &) -> ArrayBase<Expr>;

// Deduction guide from std::mdspan
template <typename T, typename Extents, typename Layout, typename Accessor>
ArrayBase(zipper::mdspan<T, Extents, Layout, Accessor>) -> ArrayBase<
    expression::nullary::MDSpan<T, Extents, Layout, Accessor>>;

UNARY_DECLARATION(ArrayBase, LogicalNot, operator!)
UNARY_DECLARATION(ArrayBase, BitNot, operator~)
UNARY_DECLARATION(ArrayBase, Negate, operator-)

SCALAR_BINARY_DECLARATION(ArrayBase, Plus, operator+)
SCALAR_BINARY_DECLARATION(ArrayBase, Minus, operator-)
SCALAR_BINARY_DECLARATION(ArrayBase, Multiplies, operator*)
SCALAR_BINARY_DECLARATION(ArrayBase, Divides, operator/)
SCALAR_BINARY_DECLARATION(ArrayBase, Modulus, operator%)
SCALAR_BINARY_DECLARATION(ArrayBase, EqualsTo, operator==)
SCALAR_BINARY_DECLARATION(ArrayBase, NotEqualsTo, operator!=)
SCALAR_BINARY_DECLARATION(ArrayBase, Greater, operator>)
SCALAR_BINARY_DECLARATION(ArrayBase, Less, operator<)
SCALAR_BINARY_DECLARATION(ArrayBase, GreaterEqual, operator>=)
SCALAR_BINARY_DECLARATION(ArrayBase, LessEqual, operator<=)
// GCC notes that these operators don't allow short circuiting, but that's ok
// for our expressions
SCALAR_BINARY_DECLARATION(ArrayBase, BitAnd, operator&)
SCALAR_BINARY_DECLARATION(ArrayBase, BitOr, operator|)
SCALAR_BINARY_DECLARATION(ArrayBase, BitXor, operator^)

BINARY_DECLARATION(ArrayBase, Plus, operator+)
BINARY_DECLARATION(ArrayBase, Minus, operator-)
BINARY_DECLARATION(ArrayBase, Multiplies, operator*)
BINARY_DECLARATION(ArrayBase, Divides, operator/)
BINARY_DECLARATION(ArrayBase, Modulus, operator%)
BINARY_DECLARATION(ArrayBase, EqualsTo, operator==)
BINARY_DECLARATION(ArrayBase, NotEqualsTo, operator!=)
BINARY_DECLARATION(ArrayBase, Greater, operator>)
BINARY_DECLARATION(ArrayBase, Less, operator<)
BINARY_DECLARATION(ArrayBase, GreaterEqual, operator>=)
BINARY_DECLARATION(ArrayBase, LessEqual, operator<=)

// GCC notes that these operators don't allow short circuiting, but that's ok
// for our expressions
BINARY_DECLARATION(ArrayBase, BitAnd, operator&)
BINARY_DECLARATION(ArrayBase, BitOr, operator|)
BINARY_DECLARATION(ArrayBase, BitXor, operator^)

BINARY_DECLARATION(ArrayBase, Min, min)
BINARY_DECLARATION(ArrayBase, Max, max)

namespace concepts::detail {
template <typename T> struct IsArray<ArrayBase<T>> : std::true_type {};
template <typename T> struct IsZipperBase<ArrayBase<T>> : std::true_type {};
} // namespace concepts::detail
} // namespace zipper

#endif
