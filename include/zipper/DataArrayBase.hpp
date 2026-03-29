#if !defined(ZIPPER_DATAARRAYBASE_HPP)
#define ZIPPER_DATAARRAYBASE_HPP

#include "ZipperBase.hpp"
#include "as.hpp"
#include "concepts/DataArray.hpp"
#include "concepts/detail/IsZipperBase.hpp"
#include "zipper/types.hpp"
//
#include "detail/extents/static_extents_to_integral_sequence.hpp"
#include "expression/nullary/Constant.hpp"
#include <compare>

namespace zipper {

namespace detail {
template <typename ValueType, concepts::Extents Extents, bool LeftMajor>
class DataArray_;
} // namespace detail

template <concepts::Expression Expr>
class DataArrayBase : public ZipperBase<DataArrayBase, Expr> {
public:
  DataArrayBase() = default;

  using expression_type = std::decay_t<Expr>;
  using expression_traits =
      expression::detail::ExpressionTraits<expression_type>;
  using value_type = typename expression_traits::value_type;
  using extents_type = typename expression_traits::extents_type;
  using extents_traits = detail::ExtentsTraits<extents_type>;
  using Base = ZipperBase<DataArrayBase, Expr>;
  using Base::Base;
  using Base::cast;
  using Base::expression;
  using Base::extent;
  using Base::extents;
  using Base::lift;
  using Base::repeat_left;
  using Base::repeat_right;
  using Base::swizzle;

  // Slice methods - delegate to ZipperBase::slice_expression
  template <typename... Slices, typename Self> auto slice(this Self&& self, Slices &&...slices) {
    auto v = std::forward<Self>(self).template slice_expression<Slices...>(
        std::forward<Slices>(slices)...);
    return DataArrayBase<std::decay_t<decltype(v)>>(std::move(v));
  }
  template <typename... Slices, typename Self> auto slice(this Self&& self) {
    auto v = std::forward<Self>(self).template slice_expression<Slices...>();
    return DataArrayBase<std::decay_t<decltype(v)>>(std::move(v));
  }

  template <index_type... N>
  auto eval(const std::integer_sequence<index_type, N...> &) const
    requires(std::is_same_v<zipper::extents<N...>, extents_type>)
  {
    return detail::DataArray_<value_type, zipper::extents<N...>, true>(
        this->expression());
  }
  auto eval() const {
    return eval(
        detail::extents::static_extents_to_integral_sequence_t<extents_type>{});
  }

  auto operator=(concepts::DataArray auto const &v) -> DataArrayBase & {
    return Base::operator=(v.expression());
  }
  auto operator=(concepts::DataArray auto &&v) -> DataArrayBase & {
    return Base::operator=(v.expression());
  }

  template <concepts::DataArray Other>
  DataArrayBase(const Other &other)
    requires(expression::concepts::WritableExpression<expression_type>)
      : DataArrayBase(other.expression()) {}

  // Catch-all forwarding constructor: routes non-DataArray arguments through
  // std::in_place so that the expression is constructed directly from the
  // forwarded args.  This enables deduction guides from std::span,
  // std::vector, std::array, and extents to work.
  template <typename... Args>
    requires(!(concepts::DataArray<std::decay_t<Args>> && ...))
  DataArrayBase(Args &&...args)
      : Base(std::in_place, std::forward<Args>(args)...) {}

  auto as_array() const { return zipper::as_array(*this); }

  /// Set all elements to the given value.
  void fill(const value_type &value)
    requires(expression::concepts::WritableExpression<expression_type>)
  {
    expression::nullary::Constant c(value, extents());
    expression().assign(c);
  }
};

template <concepts::Expression Expr>
DataArrayBase(Expr &&) -> DataArrayBase<Expr>;
template <concepts::Expression Expr>
DataArrayBase(const Expr &) -> DataArrayBase<Expr>;

namespace concepts::detail {
template <typename T> struct IsDataArray<DataArrayBase<T>> : std::true_type {};
template <typename T> struct IsZipperBase<DataArrayBase<T>> : std::true_type {};
} // namespace concepts::detail
} // namespace zipper

#endif
