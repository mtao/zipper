#if !defined(ZIPPER_FORM_HPP)
#define ZIPPER_FORM_HPP

#include "FormBase.hxx"
#include "concepts/Form.hpp"
#include "zipper/expression/nullary/MDArray.hpp"
#include "zipper/expression/nullary/MDSpan.hpp"
#include "zipper/types.hpp"
namespace zipper {

namespace detail {
template <typename ValueType, concepts::Extents Extents,
          bool LeftMajor = true>
class Form_ : public FormBase<expression::nullary::MDArray<
                  ValueType, Extents, storage::tensor_layout<LeftMajor>,
                  default_accessor_policy<ValueType>>> {
public:
  using layout_type = storage::tensor_layout<LeftMajor>;
  using expression_type =
      expression::nullary::MDArray<ValueType, Extents, layout_type,
                                   default_accessor_policy<ValueType>>;
  using Base = FormBase<expression_type>;
  using Base::expression;
  using value_type = Base::value_type;
  using extents_type = Base::extents_type;
  using span_expression_type =
      expression::nullary::MDSpan<ValueType, Extents, layout_type,
                                  default_accessor_policy<ValueType>>;
  using const_span_expression_type =
      expression::nullary::MDSpan<const ValueType, Extents, layout_type,
                                  default_accessor_policy<ValueType>>;
  using span_type = FormBase<span_expression_type>;
  using const_span_type = FormBase<const_span_expression_type>;

  using Base::Base;
  using Base::extent;
  using Base::extents;

  Form_(const Form_ &o) = default;
  Form_(Form_ &&o) = default;
  auto operator=(const Form_ &o) -> Form_ & = default;
  template <concepts::Expression Other>
  Form_(const Other &other) : Base(other) {}
  template <concepts::Expression Other>
    requires(!std::is_same_v<std::decay_t<Other>, Form_>)
  Form_(Other &&other) : Base(static_cast<const Other&>(other)) {}
  template <concepts::Form Other>
  Form_(const Other &other) : Base(other) {}
  template <concepts::Index... Args>
  Form_(Args &&...args) : Base(Extents(std::forward<Args>(args)...)) {}
  template <index_type... indices>
  Form_(const zipper::extents<indices...> &e) : Base(e) {}
  auto operator=(Form_ &&o) -> Form_ & {
    expression().operator=(std::move(o.expression()));
    return *this;
  }
  using Base::operator=;
};
} // namespace detail
template <typename ValueType, index_type... Indxs>
using Form = detail::Form_<ValueType, zipper::extents<Indxs...>>;

namespace concepts::detail {

template <typename ValueType, concepts::Extents Extents, bool LeftMajor>
struct IsForm<zipper::detail::Form_<ValueType, Extents, LeftMajor>>
    : std::true_type {};
template <typename ValueType, concepts::Extents Extents, bool LeftMajor>
struct IsZipperBase<zipper::detail::Form_<ValueType, Extents, LeftMajor>>
    : std::true_type {};

} // namespace concepts::detail
} // namespace zipper

#endif
