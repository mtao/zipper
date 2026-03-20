#if !defined(ZIPPER_ARRAYBASE_HPP)
#define ZIPPER_ARRAYBASE_HPP

#include "ZipperBase.hpp"
#include "concepts/detail/IsZipperBase.hpp"
#include "concepts/stl.hpp"
#include "expression/nullary/StlMDArray.hpp"
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
#include "expression/unary/Abs.hpp"
#include "expression/unary/ScalarPower.hpp"

namespace zipper {

namespace detail {
template <typename ValueType, concepts::Extents Extents, bool LeftMajor>
class Array_;
} // namespace detail

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

  template <typename... Args>
    requires(!(concepts::Array<Args> && ...))
  ArrayBase(Args &&...args)
      : Base(std::in_place, std::forward<Args>(args)...) {}

  template <index_type... N>
  auto eval(const std::integer_sequence<index_type, N...> &) const
    requires(std::is_same_v<extents<N...>, extents_type>)
  {
    return detail::Array_<value_type, zipper::extents<N...>, true>(this->expression());
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
    requires(expression::concepts::WritableExpression<expression_type>)
      : ArrayBase(other.expression()) {}

  auto operator*=(const value_type &other) -> ArrayBase &
    requires(expression::concepts::WritableExpression<expression_type>)
  {
    return *this = other * *this;
  }
  auto operator/=(const value_type &other) -> ArrayBase &
    requires(expression::concepts::WritableExpression<expression_type>)
  {
    return *this = *this / other;
  }
  template <concepts::Array Other>
  auto operator+=(const Other &other) -> ArrayBase &
    requires(expression::concepts::WritableExpression<expression_type>)
  {
    return *this = *this + other;
  }
  template <concepts::Array Other>
  auto operator-=(const Other &other) -> ArrayBase &
    requires(expression::concepts::WritableExpression<expression_type>)
  {
    return *this = *this - other;
  }
  template <concepts::Array Other>
  auto operator*=(const Other &other) -> ArrayBase &
    requires(expression::concepts::WritableExpression<expression_type>)
  {
    return *this = *this * other;
  }
  template <concepts::Array Other>
  auto operator/=(const Other &other) -> ArrayBase &
    requires(expression::concepts::WritableExpression<expression_type>)
  {
    return *this = *this / other;
  }

  //--------------------------------------------------------------
  //

  template <typename Self>
  auto pow(this Self&& self, value_type const &exp) {
    using child_t = detail::member_child_storage_t<Self, expression_type>;
    return ArrayBase<
        expression::unary::ScalarPower<child_t, value_type>>(
        std::in_place, std::forward<Self>(self).expression(), exp);
  }

  template <typename Self>
  auto abs(this Self&& self) {
    using child_t = detail::member_child_storage_t<Self, expression_type>;
    return ArrayBase<expression::unary::Abs<child_t>>(
        std::in_place, std::forward<Self>(self).expression());
  }

  auto sum() const -> value_type {
    return expression::reductions::CoefficientSum{expression()}();
  }

  auto product() const -> value_type {
    return expression::reductions::CoefficientProduct{expression()}();
  }

  template <index_type T> auto norm_powered() const -> value_type {
    return expression::reductions::LpNormPowered<T, const expression_type &>(
        expression())();
  }
  auto norm_powered(value_type T) const -> value_type {
    return pow(T).abs().sum();
  }

  template <index_type T = 2> auto norm() const -> value_type {
    return expression::reductions::LpNorm<T, const expression_type &>(
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

  // Slice methods - construct wrapper in-place to avoid moving non-movable expressions
  template <typename... Slices, typename Self> auto slice(this Self&& self, Slices &&...slices) {
    using child_t = detail::member_child_storage_t<Self, expression_type>;
    using V = expression::unary::Slice<child_t,
                  detail::slice_type_for_t<std::decay_t<Slices>>...>;
    return ArrayBase<V>(std::in_place, std::forward<Self>(self).expression(),
        Base::filter_args_for_zipperbase(std::forward<Slices>(slices))...);
  }
  template <typename... Slices, typename Self> auto slice(this Self&& self) {
    using child_t = detail::member_child_storage_t<Self, expression_type>;
    using V = expression::unary::Slice<child_t, std::decay_t<Slices>...>;
    return ArrayBase<V>(std::in_place, std::forward<Self>(self).expression(), Slices{}...);
  }
};

template <concepts::Expression Expr>
ArrayBase(Expr &&) -> ArrayBase<Expr>;
template <concepts::Expression Expr>
ArrayBase(const Expr &) -> ArrayBase<Expr>;

// STL deduction guides: rvalue → owning StlMDArray, lvalue → borrowing StlMDArray
template <concepts::StlStorage S>
ArrayBase(S &&) -> ArrayBase<expression::nullary::StlMDArray<std::decay_t<S>>>;
template <concepts::StlStorage S>
ArrayBase(S &) -> ArrayBase<expression::nullary::StlMDArray<S &>>;

namespace concepts::detail {
template <typename T> struct IsArray<ArrayBase<T>> : std::true_type {};
template <typename T> struct IsZipperBase<ArrayBase<T>> : std::true_type {};
} // namespace concepts::detail
} // namespace zipper

#endif
