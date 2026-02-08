#if !defined(ZIPPER_FORMBASE_HPP)
#define ZIPPER_FORMBASE_HPP

#include "ZipperBase.hpp"
#include "TensorBase.hpp"
#include "as.hpp"
#include "concepts/Form.hpp"
#include "concepts/Tensor.hpp"
#include "concepts/Vector.hpp"
#include "concepts/detail/IsZipperBase.hpp"
#include "detail/extents/static_extents_to_integral_sequence.hpp"
#include "expression/nullary/MDSpan.hpp"
#include "zipper/detail/declare_operations.hpp"
#include "zipper/expression/binary/ArithmeticExpressions.hpp"
#include "zipper/expression/binary/FormTensorProduct.hpp"
#include "zipper/expression/binary/WedgeProduct.hpp"
#include "zipper/expression/unary/ScalarArithmetic.hpp"

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
    requires(expression_traits::is_writable)
      : FormBase(other.expression()) {}
  template <concepts::Form Other>
  auto operator=(const Other &other) -> FormBase &
    requires(expression_traits::is_writable)
  {
    return operator=(other.expression());
  }

  template <typename... Slices> auto slice() {
    auto v = Base::template slice_expression<Slices...>();
    return FormBase<std::decay_t<decltype(v)>>(std::move(v));
  }

  template <typename... Slices> auto slice(Slices &&...slices) const {
    auto v =
        Base::template slice_expression<Slices...>(std::forward<Slices>(slices)...);
    return FormBase<std::decay_t<decltype(v)>>(std::move(v));
  }
  template <typename... Slices> auto slice() const {
    auto v = Base::template slice_expression<Slices...>();
    return FormBase<std::decay_t<decltype(v)>>(std::move(v));
  }

  template <typename... Slices> auto slice(Slices &&...slices) {
    auto v =
        Base::template slice_expression<Slices...>(std::forward<Slices>(slices)...);
    return FormBase<std::decay_t<decltype(v)>>(std::move(v));
  }

  auto operator*() const {
    // TODO: this needs to be implemented
    assert(false);
    return *this;
  }
};

template <concepts::Expression Expr>
FormBase(Expr &&) -> FormBase<Expr>;
template <concepts::Expression Expr>
FormBase(const Expr &) -> FormBase<Expr>;

// Deduction guide from std::mdspan
template <typename T, typename Extents, typename Layout, typename Accessor>
FormBase(zipper::mdspan<T, Extents, Layout, Accessor>) -> FormBase<
    expression::nullary::MDSpan<T, Extents, Layout, Accessor>>;

UNARY_DECLARATION(FormBase, LogicalNot, operator!)
UNARY_DECLARATION(FormBase, BitNot, operator~)
UNARY_DECLARATION(FormBase, Negate, operator-)

SCALAR_BINARY_DECLARATION(FormBase, Plus, operator+)
SCALAR_BINARY_DECLARATION(FormBase, Minus, operator-)
SCALAR_BINARY_DECLARATION(FormBase, Multiplies, operator*)
SCALAR_BINARY_DECLARATION(FormBase, Divides, operator/)

BINARY_DECLARATION(FormBase, Plus, operator+)
BINARY_DECLARATION(FormBase, Minus, operator-)

template <concepts::Form Expr1, concepts::Form Expr2>
auto operator^(Expr1 const &lhs, Expr2 const &rhs) {
  using V = expression::binary::WedgeProduct<const typename Expr1::expression_type,
                                            const typename Expr2::expression_type>;
  return FormBase<V>(V(lhs.expression(), rhs.expression()));
}

template <concepts::Form Expr1, concepts::Tensor Expr2>
auto operator*(Expr1 const &lhs, Expr2 const &rhs) {
  using V = expression::binary::FormTensorProduct<const typename Expr1::expression_type,
                                                  const typename Expr2::expression_type>;

  if constexpr (V::result_is_form) {
    return FormBase<V>(V(lhs.expression(), rhs.expression()));
  } else {
    return TensorBase<V>(V(lhs.expression(), rhs.expression()));
  }
}
template <concepts::Form Expr1, concepts::Vector Expr2>
auto operator*(Expr1 const &lhs, Expr2 const &rhs) {

  using V = expression::binary::FormTensorProduct<const typename Expr1::expression_type,
                                                  const typename Expr2::expression_type>;
  if constexpr (V::result_is_form) {
    return FormBase<V>(V(lhs.expression(), rhs.expression()));
  } else {
    return TensorBase<V>(V(lhs.expression(), rhs.expression()));
  }
}

namespace concepts::detail {
template <typename T>
struct IsForm<FormBase<T>> : std::true_type {};
template <typename T>
struct IsZipperBase<FormBase<T>> : std::true_type {};
} // namespace concepts::detail

} // namespace zipper

#endif
