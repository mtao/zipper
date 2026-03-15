#if !defined(ZIPPER_FORMBASE_HPP)
#define ZIPPER_FORMBASE_HPP

#include "ZipperBase.hpp"
#include <cassert>
#include "as.hpp"
#include "concepts/Form.hpp"
#include "concepts/Tensor.hpp"
#include "concepts/Vector.hpp"
#include "concepts/detail/IsZipperBase.hpp"
#include "concepts/stl.hpp"
#include "detail/extents/static_extents_to_integral_sequence.hpp"
#include "expression/nullary/StlMDArray.hpp"
#include <utility>

namespace zipper {

namespace detail {
template <typename ValueType, concepts::Extents Extents, bool LeftMajor>
class Form_;
} // namespace detail

template <concepts::Expression Expr>
class FormBase : public ZipperBase<FormBase, Expr> {
public:
  using Base = ZipperBase<FormBase, Expr>;
  FormBase() = default;

  using expression_type = typename Base::expression_type;
  using expression_traits = typename Base::expression_traits;
  using value_type = typename expression_traits::value_type;
  using extents_type = typename expression_traits::extents_type;
  using extents_traits = detail::ExtentsTraits<extents_type>;

  auto as_array() const { return zipper::as_array(*this); }
  auto as_tensor() const { return zipper::as_tensor(*this); }
  auto as_vector() const { return zipper::as_vector(*this); }
  template <index_type... N>
  auto eval(const std::integer_sequence<index_type, N...> &) const
    requires(std::is_same_v<extents<N...>, extents_type>)
  {
    return detail::Form_<value_type, extents<N...>, true>(this->expression());
  }
  auto eval() const {
    return eval(
        detail::extents::static_extents_to_integral_sequence_t<extents_type>{});
  }

  using Base::Base;
  using Base::operator=;
  using Base::cast;
  using Base::expression;
  using Base::swizzle;
  FormBase(FormBase &&) = default;
  FormBase(const FormBase &) = default;

  template <typename... Args>
    requires(!(concepts::Form<Args> && ...))
  FormBase(Args &&...args)
      : Base(std::in_place, std::forward<Args>(args)...) {}

  auto operator=(concepts::Form auto const &v) -> FormBase & {
    expression() = v.expression();
    return *this;
  }
  auto operator=(concepts::Form auto &&v) -> FormBase & {
    expression() = v.expression();
    return *this;
  }
  auto operator=(const FormBase &v) -> FormBase & {
    Base::operator=(v.expression());
    return *this;
  }
  auto operator=(FormBase &&v) -> FormBase & {
    Base::operator=(v.expression());
    return *this;
  }

  template <concepts::Form Other>
  FormBase(const Other &other)
    requires(expression::concepts::WritableExpression<expression_type>)
      : FormBase(other.expression()) {}
  template <concepts::Form Other>
  auto operator=(const Other &other) -> FormBase &
    requires(expression::concepts::WritableExpression<expression_type>)
  {
    return operator=(other.expression());
  }

  template <typename... Slices> auto slice() {
    using V = expression::unary::Slice<expression_type&, std::decay_t<Slices>...>;
    return FormBase<V>(std::in_place, expression(), Slices{}...);
  }

  template <typename... Slices> auto slice(Slices &&...slices) const {
    using V = expression::unary::Slice<const expression_type&,
                  detail::slice_type_for_t<std::decay_t<Slices>>...>;
    return FormBase<V>(std::in_place, expression(),
        Base::filter_args_for_zipperbase(std::forward<Slices>(slices))...);
  }
  template <typename... Slices> auto slice() const {
    using V = expression::unary::Slice<const expression_type&, std::decay_t<Slices>...>;
    return FormBase<V>(std::in_place, expression(), Slices{}...);
  }

  template <typename... Slices> auto slice(Slices &&...slices) {
    using V = expression::unary::Slice<expression_type&,
                  detail::slice_type_for_t<std::decay_t<Slices>>...>;
    return FormBase<V>(std::in_place, expression(),
        Base::filter_args_for_zipperbase(std::forward<Slices>(slices))...);
  }

  auto operator*() const {
    // TODO: this needs to be implemented
    std::unreachable();
    return *this;
  }
};

template <concepts::Expression Expr>
FormBase(Expr &&) -> FormBase<Expr>;
template <concepts::Expression Expr>
FormBase(const Expr &) -> FormBase<Expr>;

// STL deduction guides: rvalue → owning StlMDArray, lvalue → borrowing StlMDArray
template <concepts::StlStorage S>
FormBase(S &&) -> FormBase<expression::nullary::StlMDArray<std::decay_t<S>>>;
template <concepts::StlStorage S>
FormBase(S &) -> FormBase<expression::nullary::StlMDArray<S &>>;

namespace concepts::detail {
template <typename T>
struct IsForm<FormBase<T>> : std::true_type {};
template <typename T>
struct IsZipperBase<FormBase<T>> : std::true_type {};
} // namespace concepts::detail

} // namespace zipper

#endif
