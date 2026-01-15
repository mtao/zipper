#if !defined(ZIPPER_ARRAYBASE_HPP)
#define ZIPPER_ARRAYBASE_HPP

#include "ZipperBase.hpp"
#include "views/reductions/All.hpp"
#include "views/reductions/Any.hpp"
#include "views/reductions/CoefficientProduct.hpp"
#include "views/reductions/CoefficientSum.hpp"
#include "views/reductions/LpNorm.hpp"
#include "views/reductions/LpNormPowered.hpp"
#include "zipper/types.hpp"
//
#include "concepts/ArrayBaseDerived.hpp"
#include "concepts/ViewDerived.hpp"
//
////
#include "detail/extents/static_extents_to_integral_sequence.hpp"
#include "views/unary/AbsView.hpp"
#include "views/unary/ScalarPowerView.hpp"
#include "views/unary/detail/operation_implementations.hpp"
#include "zipper/detail/declare_operations.hpp"
#include "zipper/views/binary/ArithmeticViews.hpp"
#include "zipper/views/unary/ScalarArithmeticViews.hpp"

namespace zipper {

template <concepts::QualifiedViewDerived View>
class ArrayBase : public ZipperBase<ArrayBase, View> {
public:
  ArrayBase() = default;

  using view_type = View;
  using traits = views::detail::ViewTraits<View>;
  using value_type = typename traits::value_type;
  using extents_type = typename traits::extents_type;
  using extents_traits = detail::ExtentsTraits<extents_type>;
  using Base = ZipperBase<ArrayBase, View>;
  using Base::Base;
  using Base::cast;
  using Base::swizzle;
  using Base::view;

  template <index_type... N>
  auto eval(const std::integer_sequence<index_type, N...> &) const
    requires(std::is_same_v<extents<N...>, extents_type>)
  {
    return Array_<value_type, zipper::extents<N...>>(this->view());
  }
  auto eval() const {
    return eval(
        detail::extents::static_extents_to_integral_sequence_t<extents_type>{});
  }
  auto operator=(concepts::ArrayBaseDerived auto const &v) -> ArrayBase & {
    return Base::operator=(v.view());
  }
  auto operator=(concepts::ArrayBaseDerived auto &&v) -> ArrayBase & {
    return Base::operator=(v.view());
  }

  template <concepts::ArrayBaseDerived Other>
  ArrayBase(const Other &other)
    requires(view_type::is_writable)
      : ArrayBase(other.view()) {}

  auto operator*=(const value_type &other) -> ArrayBase &
    requires(view_type::is_writable)
  {
    return *this = other * *this;
  }
  auto operator/=(const value_type &other) -> ArrayBase &
    requires(view_type::is_writable)
  {
    return *this = *this / other;
  }
  template <concepts::ArrayBaseDerived Other>
  auto operator+=(const Other &other) -> ArrayBase &
    requires(view_type::is_writable)
  {
    return *this = *this + other;
  }
  template <concepts::ArrayBaseDerived Other>
  auto operator-=(const Other &other) -> ArrayBase &
    requires(view_type::is_writable)
  {
    return *this = *this - other;
  }
  template <concepts::ArrayBaseDerived Other>
  auto operator*=(const Other &other) -> ArrayBase &
    requires(view_type::is_writable)
  {
    return *this = *this * other;
  }
  template <concepts::ArrayBaseDerived Other>
  auto operator/=(const Other &other) -> ArrayBase &
    requires(view_type::is_writable)
  {
    return *this = *this / other;
  }

  //--------------------------------------------------------------
  //

  auto pow(value_type const &exp) const {
    return ArrayBase<
        views::unary::ScalarPowerView<const view_type, value_type>>(view(),
                                                                    exp);
  }

  auto abs() const {
    return ArrayBase<views::unary::AbsView<const view_type>>(view());
  }

  auto sum() const -> value_type {
    return views::reductions::CoefficientSum{view()}();
  }

  auto product() const -> value_type {
    return views::reductions::CoefficientProduct{view()}();
  }

  template <index_type T> auto norm_powered() const -> value_type {
    return views::reductions::LpNormPowered<T, const view_type>(view())();
  }
  auto norm_powered(value_type T) const -> value_type {
    return pow(T).abs().sum();
  }

  template <index_type T = 2> auto norm() const -> value_type {
    return views::reductions::LpNorm<T, const view_type>(view())();
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
    return views::reductions::Any(view())();
  }
  [[nodiscard]] auto all() const -> bool
    requires(std::is_same_v<value_type, bool>)
  {
    return views::reductions::All(view())();
  }
  template <typename... Slices> auto slice() {
    auto v = Base::template slice_view<Slices...>();
    return ArrayBase<std::decay_t<decltype(v)>>(std::move(v));
  }

  template <typename... Slices> auto slice(Slices &&...slices) const {
    auto v =
        Base::template slice_view<Slices...>(std::forward<Slices>(slices)...);
    return ArrayBase<std::decay_t<decltype(v)>>(std::move(v));
  }
  template <typename... Slices> auto slice() const {
    auto v = Base::template slice_view<Slices...>();
    return ArrayBase<std::decay_t<decltype(v)>>(std::move(v));
  }

  template <typename... Slices> auto slice(Slices &&...slices) {
    auto v =
        Base::template slice_view<Slices...>(std::forward<Slices>(slices)...);
    return ArrayBase<std::decay_t<decltype(v)>>(std::move(v));
  }
};

template <concepts::QualifiedViewDerived View>
ArrayBase(View &&view) -> ArrayBase<View>;
template <concepts::QualifiedViewDerived View>
ArrayBase(const View &view) -> ArrayBase<View>;
template <class T, std::size_t Size = std::dynamic_extent>
ArrayBase(std::span<T, Size> s)
    -> ArrayBase<storage::SpanStorage<T, zipper::extents<Size>>>;
template <class T, std::size_t Size = std::dynamic_extent>
ArrayBase(std::span<const T, Size> s)
    -> ArrayBase<storage::SpanStorage<T, zipper::extents<Size>>>;

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
// for our views
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
SCALAR_BINARY_DECLARATION(ArrayBase, LogicalAnd, operator&&)
SCALAR_BINARY_DECLARATION(ArrayBase, LogicalOr, operator||)
#pragma GCC diagnostic pop
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
// for our views
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
BINARY_DECLARATION(ArrayBase, LogicalAnd, operator&&)
BINARY_DECLARATION(ArrayBase, LogicalOr, operator||)
#pragma GCC diagnostic pop
BINARY_DECLARATION(ArrayBase, BitAnd, operator&)
BINARY_DECLARATION(ArrayBase, BitOr, operator|)
BINARY_DECLARATION(ArrayBase, BitXor, operator^)

BINARY_DECLARATION(ArrayBase, Min, min)
BINARY_DECLARATION(ArrayBase, Max, max)

namespace concepts::detail {
template <typename T> struct ArrayBaseDerived<ArrayBase<T>> : std::true_type {};
} // namespace concepts::detail
} // namespace zipper

#endif
