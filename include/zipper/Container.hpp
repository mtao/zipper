#if !defined(ZIPPER_CONTAINER_HPP)
#define ZIPPER_CONTAINER_HPP

#include "ContainerBase.hxx"
#include "concepts/Container.hpp"
#include "concepts/Extents.hpp"
#include "zipper/expression/nullary/MDArray.hpp"
#include "zipper/expression/nullary/MDSpan.hpp"
#include "zipper/types.hpp"

namespace zipper {

namespace detail {

template <typename ValueType, concepts::Extents Extents,
          bool LeftMajor = true>
class Container_
    : public ContainerBase<expression::nullary::MDArray<
          ValueType, Extents, storage::tensor_layout<LeftMajor>,
          default_accessor_policy<ValueType>>> {
public:
  using layout_type = storage::tensor_layout<LeftMajor>;
  using expression_type =
      expression::nullary::MDArray<ValueType, Extents, layout_type,
                                   default_accessor_policy<ValueType>>;
  using Base = ContainerBase<expression_type>;
  using Base::expression;
  using value_type = Base::value_type;
  using extents_type = Base::extents_type;
  using Base::extent;
  using Base::extents;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
  constexpr static bool is_static = extents_traits::is_static;

  using span_expression_type =
      expression::nullary::MDSpan<ValueType, Extents, layout_type,
                                  default_accessor_policy<ValueType>>;
  using const_span_expression_type =
      expression::nullary::MDSpan<const ValueType, Extents, layout_type,
                                  default_accessor_policy<ValueType>>;
  using span_type = ContainerBase<span_expression_type>;
  using const_span_type = ContainerBase<const_span_expression_type>;

  Container_() = default;
  Container_(const Container_ &o) = default;
  Container_(Container_ &&o) = default;
  auto operator=(const Container_ &o) -> Container_ & = default;
  auto operator=(Container_ &&o) -> Container_ & {
    expression().operator=(std::move(o.expression()));
    return *this;
  }

  template <concepts::Expression Other>
  Container_(const Other &other) : Base(other) {}

  template <concepts::Container Other>
  Container_(const Other &other) : Base(other) {}

  template <typename... Args>
  Container_(Args &&...args)
    requires((std::is_convertible_v<Args, index_type> && ...))
      : Base(Extents(std::forward<Args>(args)...)) {}

  template <index_type... indices>
  Container_(const zipper::extents<indices...> &e) : Base(e) {}

  auto as_span() -> span_type {
    if constexpr (is_static) {
      return span_type(expression().as_std_span());
    } else {
      return span_type(expression().as_std_span(), extents());
    }
  }
  auto as_const_span() const -> const_span_type {
    if constexpr (is_static) {
      return const_span_type(expression().as_std_span());
    } else {
      return const_span_type(expression().as_std_span(), extents());
    }
  }
  auto as_span() const -> const_span_type { return as_const_span(); }

  auto begin() { return expression().linear_accessor().begin(); }
  auto end() { return expression().linear_accessor().end(); }
  auto begin() const { return expression().linear_accessor().begin(); }
  auto end() const { return expression().linear_accessor().end(); }

  auto data() { return expression().data(); }
  auto data() const { return expression().data(); }

  using Base::operator=;
};

template <concepts::Container CB>
Container_(const CB &o)
    -> Container_<typename CB::value_type, typename CB::extents_type>;

} // namespace detail

template <typename ValueType, index_type... Indxs>
using Container = detail::Container_<ValueType, zipper::extents<Indxs...>>;

namespace concepts::detail {

template <typename ValueType, concepts::Extents Extents, bool LeftMajor>
struct IsContainer<zipper::detail::Container_<ValueType, Extents, LeftMajor>>
    : std::true_type {};
template <typename ValueType, concepts::Extents Extents, bool LeftMajor>
struct IsZipperBase<zipper::detail::Container_<ValueType, Extents, LeftMajor>>
    : std::true_type {};

} // namespace concepts::detail

} // namespace zipper
#endif
