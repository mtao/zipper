#if !defined(ZIPPER_CONTAINERBASE_HPP)
#define ZIPPER_CONTAINERBASE_HPP

#include "ZipperBase.hpp"
#include "as.hpp"
#include "concepts/Container.hpp"
#include "concepts/detail/IsZipperBase.hpp"
#include "zipper/types.hpp"
//
#include "detail/extents/static_extents_to_integral_sequence.hpp"
#include <compare>

namespace zipper {

namespace detail {
template <typename ValueType, concepts::Extents Extents, bool LeftMajor>
class Container_;
} // namespace detail

template <concepts::Expression Expr>
class ContainerBase : public ZipperBase<ContainerBase, Expr> {
public:
  ContainerBase() = default;

  using expression_type = std::decay_t<Expr>;
  using expression_traits =
      expression::detail::ExpressionTraits<expression_type>;
  using value_type = typename expression_traits::value_type;
  using extents_type = typename expression_traits::extents_type;
  using extents_traits = detail::ExtentsTraits<extents_type>;
  using Base = ZipperBase<ContainerBase, Expr>;
  using Base::Base;
  using Base::cast;
  using Base::expression;
  using Base::extent;
  using Base::extents;
  using Base::repeat_left;
  using Base::repeat_right;
  using Base::swizzle;

  // Slice methods - delegate to ZipperBase::slice_expression
  template <typename... Slices> auto slice() {
    auto v = Base::template slice_expression<Slices...>();
    return ContainerBase<std::decay_t<decltype(v)>>(std::move(v));
  }
  template <typename... Slices> auto slice(Slices &&...slices) const {
    auto v = Base::template slice_expression<Slices...>(
        std::forward<Slices>(slices)...);
    return ContainerBase<std::decay_t<decltype(v)>>(std::move(v));
  }
  template <typename... Slices> auto slice() const {
    auto v = Base::template slice_expression<Slices...>();
    return ContainerBase<std::decay_t<decltype(v)>>(std::move(v));
  }
  template <typename... Slices> auto slice(Slices &&...slices) {
    auto v = Base::template slice_expression<Slices...>(
        std::forward<Slices>(slices)...);
    return ContainerBase<std::decay_t<decltype(v)>>(std::move(v));
  }

  template <index_type... N>
  auto eval(const std::integer_sequence<index_type, N...> &) const
    requires(std::is_same_v<zipper::extents<N...>, extents_type>)
  {
    return detail::Container_<value_type, zipper::extents<N...>, true>(
        this->expression());
  }
  auto eval() const {
    return eval(
        detail::extents::static_extents_to_integral_sequence_t<extents_type>{});
  }

  auto operator=(concepts::Container auto const &v) -> ContainerBase & {
    return Base::operator=(v.expression());
  }
  auto operator=(concepts::Container auto &&v) -> ContainerBase & {
    return Base::operator=(v.expression());
  }

  template <concepts::Container Other>
  ContainerBase(const Other &other)
    requires(expression_traits::is_writable)
      : ContainerBase(other.expression()) {}

  auto as_array() const { return zipper::as_array(*this); }
};

template <concepts::Expression Expr>
ContainerBase(Expr &&) -> ContainerBase<Expr>;
template <concepts::Expression Expr>
ContainerBase(const Expr &) -> ContainerBase<Expr>;

namespace concepts::detail {
template <typename T> struct IsContainer<ContainerBase<T>> : std::true_type {};
template <typename T> struct IsZipperBase<ContainerBase<T>> : std::true_type {};
} // namespace concepts::detail
} // namespace zipper

#endif
