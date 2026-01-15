#if !defined(ZIPPER_FORMBASE_HPP)
#define ZIPPER_FORMBASE_HPP

#include "ZipperBase.hpp"
#include "as.hpp"
#include "concepts/FormBaseDerived.hpp"
#include "concepts/TensorBaseDerived.hpp"
#include "concepts/VectorBaseDerived.hpp"
#include "detail/extents/static_extents_to_integral_sequence.hpp"
#include "zipper/views/binary/FormTensorProductView.hpp"
#include "zipper/views/binary/WedgeProductView.hpp"

namespace zipper {

template <concepts::QualifiedViewDerived View>
class FormBase : public ZipperBase<FormBase, View> {
public:
  using Base = ZipperBase<FormBase, View>;
  FormBase() = default;

  using view_type = View;
  using value_type = View::value_type;
  using extents_type = View::extents_type;
  using extents_traits = detail::ExtentsTraits<extents_type>;

  auto as_array() const { return zipper::as_array(*this); }
  auto as_tensor() const { return zipper::as_tensor(*this); }
  auto as_vector() const { return zipper::as_vector(*this); }
  template <index_type... N>
  auto eval(const std::integer_sequence<index_type, N...> &) const
    requires(std::is_same_v<extents<N...>, extents_type>)
  {
    return Form_<value_type, extents<N...>>(this->view());
  }
  auto eval() const {
    return eval(
        detail::extents::static_extents_to_integral_sequence_t<extents_type>{});
  }

  using Base::Base;
  using Base::operator=;
  using Base::cast;
  using Base::swizzle;
  using Base::view;
  FormBase(FormBase &&) = default;
  FormBase(const FormBase &) = default;

  auto operator=(concepts::FormBaseDerived auto const &v) -> FormBase & {
    view() = v.view();
    return *this;
  }
  auto operator=(concepts::FormBaseDerived auto &&v) -> FormBase & {
    view() = v.view();
    return *this;
  }
  auto operator=(const FormBase &v) -> FormBase & {
    Base::operator=(v.view());
    return *this;
  }
  auto operator=(FormBase &&v) -> FormBase & {
    Base::operator=(v.view());
    return *this;
  }

  template <concepts::FormBaseDerived Other>
  FormBase(const Other &other)
    requires(view_type::is_writable)
      : FormBase(other.view()) {}
  template <concepts::FormBaseDerived Other>
  auto operator=(const Other &other) -> FormBase &
    requires(view_type::is_writable)
  {
    return operator=(other.view());
  }

  template <typename... Slices> auto slice() {
    auto v = Base::template slice_view<Slices...>();
    return FormBase<std::decay_t<decltype(v)>>(std::move(v));
  }

  template <typename... Slices> auto slice(Slices &&...slices) const {
    auto v =
        Base::template slice_view<Slices...>(std::forward<Slices>(slices)...);
    return FormBase<std::decay_t<decltype(v)>>(std::move(v));
  }
  template <typename... Slices> auto slice() const {
    auto v = Base::template slice_view<Slices...>();
    return FormBase<std::decay_t<decltype(v)>>(std::move(v));
  }

  template <typename... Slices> auto slice(Slices &&...slices) {
    auto v =
        Base::template slice_view<Slices...>(std::forward<Slices>(slices)...);
    return FormBase<std::decay_t<decltype(v)>>(std::move(v));
  }

  auto operator*() const {
    // TODO: this needs to be implemented
    assert(false);
    return *this;
    // using V = views::binary::WedgeProductView<typename View1::view_type,
    //                                           typename View2::view_type>;
    // return FormBase<V>(V(lhs.view(), rhs.view()));
  }
};

template <concepts::QualifiedViewDerived View>
FormBase(View &&view) -> FormBase<View>;
template <concepts::QualifiedViewDerived View>
FormBase(const View &view) -> FormBase<View>;
template <class T, std::size_t Size = std::dynamic_extent>
FormBase(const std::span<T, Size> &s)
    -> FormBase<storage::SpanStorage<T, zipper::extents<Size>>>;

UNARY_DECLARATION(FormBase, LogicalNot, operator!)
UNARY_DECLARATION(FormBase, BitNot, operator~)
UNARY_DECLARATION(FormBase, Negate, operator-)

SCALAR_BINARY_DECLARATION(FormBase, Plus, operator+)
SCALAR_BINARY_DECLARATION(FormBase, Minus, operator-)
SCALAR_BINARY_DECLARATION(FormBase, Multiplies, operator*)
SCALAR_BINARY_DECLARATION(FormBase, Divides, operator/)

BINARY_DECLARATION(FormBase, Plus, operator+)
BINARY_DECLARATION(FormBase, Minus, operator-)

template <concepts::FormBaseDerived View1, concepts::FormBaseDerived View2>
auto operator^(View1 const &lhs, View2 const &rhs) {
  using V = views::binary::WedgeProductView<typename View1::view_type,
                                            typename View2::view_type>;
  return FormBase<V>(V(lhs.view(), rhs.view()));
}

template <concepts::FormBaseDerived View1, concepts::TensorBaseDerived View2>
auto operator*(View1 const &lhs, View2 const &rhs) {
  using V = views::binary::FormTensorProductView<typename View1::view_type,
                                                 typename View2::view_type>;

  return FormBase<V>(V(lhs.view(), rhs.view()));
}
template <concepts::FormBaseDerived View1, concepts::VectorBaseDerived View2>
auto operator*(View1 const &lhs, View2 const &rhs) {

  using V = views::binary::FormTensorProductView<typename View1::view_type,
                                                 typename View2::view_type>;
  return FormBase<V>(V(lhs.view(), rhs.view()));
}
namespace concepts::detail {}

} // namespace zipper

#endif
