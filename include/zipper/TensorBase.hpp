#if !defined(ZIPPER_TENSORBASE_HPP)
#define ZIPPER_TENSORBASE_HPP

#include "ZipperBase.hpp"
#include "concepts/Tensor.hpp"
#include "detail/extents/static_extents_to_integral_sequence.hpp"
#include "expression/nullary/MDSpan.hpp"
#include "zipper/detail/declare_operations.hpp"
#include "zipper/expression/binary/ArithmeticExpressions.hpp"
#include "zipper/expression/binary/TensorProduct.hpp"
#include "zipper/expression/unary/ScalarArithmetic.hpp"

namespace zipper {

template <concepts::Expression Expr>
class TensorBase : public ZipperBase<TensorBase, Expr> {
public:
  using Base = ZipperBase<TensorBase, Expr>;
  TensorBase() = default;

  using expression_type = typename Base::expression_type;
  using expression_traits = typename Base::expression_traits;
  using value_type = expression_traits::value_type;
  using extents_type = expression_traits::extents_type;
  using extents_traits = detail::ExtentsTraits<extents_type>;

  template <index_type... N>
  auto eval(const std::integer_sequence<index_type, N...> &) const
    requires(std::is_same_v<extents<N...>, extents_type>)
  {
    return Tensor_<value_type, extents<N...>>(this->expression());
  }
  auto eval() const {
    return eval(
        detail::extents::static_extents_to_integral_sequence_t<extents_type>{});
  }

  using Base::Base;
  using Base::operator=;
  using Base::expression;

  auto operator=(concepts::Tensor auto const &v) -> TensorBase & {
    expression() = v.expression();
    return *this;
  }

  template <concepts::Tensor Other>
  TensorBase(const Other &other)
    requires(expression_traits::is_writable)
      : TensorBase(other.expression()) {}
  template <concepts::Tensor Other>
  auto operator=(const Other &other) -> TensorBase &
    requires(expression_traits::is_writable)
  {
    return operator=(other.expression());
  }
  auto operator=(concepts::Tensor auto &&v) -> TensorBase & {
    return Base::operator=(v.expression());
  }

  // Slice methods - delegate to ZipperBase::slice_expression
  template <typename... Slices> auto slice() {
    auto v = Base::template slice_expression<Slices...>();
    return TensorBase<std::decay_t<decltype(v)>>(std::move(v));
  }
  template <typename... Slices> auto slice(Slices &&...slices) const {
    auto v =
        Base::template slice_expression<Slices...>(std::forward<Slices>(slices)...);
    return TensorBase<std::decay_t<decltype(v)>>(std::move(v));
  }
  template <typename... Slices> auto slice() const {
    auto v = Base::template slice_expression<Slices...>();
    return TensorBase<std::decay_t<decltype(v)>>(std::move(v));
  }
  template <typename... Slices> auto slice(Slices &&...slices) {
    auto v =
        Base::template slice_expression<Slices...>(std::forward<Slices>(slices)...);
    return TensorBase<std::decay_t<decltype(v)>>(std::move(v));
  }
};

template <concepts::Expression Expr>
TensorBase(Expr &&) -> TensorBase<Expr>;
template <concepts::Expression Expr>
TensorBase(const Expr &) -> TensorBase<Expr>;

// Deduction guide from std::mdspan
template <typename T, typename Extents, typename Layout, typename Accessor>
TensorBase(zipper::mdspan<T, Extents, Layout, Accessor>) -> TensorBase<
    expression::nullary::MDSpan<T, Extents, Layout, Accessor>>;

UNARY_DECLARATION(TensorBase, LogicalNot, operator!)
UNARY_DECLARATION(TensorBase, BitNot, operator~)
UNARY_DECLARATION(TensorBase, Negate, operator-)

SCALAR_BINARY_DECLARATION(TensorBase, Plus, operator+)
SCALAR_BINARY_DECLARATION(TensorBase, Minus, operator-)
SCALAR_BINARY_DECLARATION(TensorBase, Multiplies, operator*)
SCALAR_BINARY_DECLARATION(TensorBase, Divides, operator/)

BINARY_DECLARATION(TensorBase, Plus, operator+)
BINARY_DECLARATION(TensorBase, Minus, operator-)

template <concepts::Tensor Expr1, concepts::Tensor Expr2>
auto operator*(Expr1 const &lhs, Expr2 const &rhs) {
  using V = expression::binary::TensorProduct<const typename Expr1::expression_type,
                                              const typename Expr2::expression_type>;
  return TensorBase<V>(V(lhs.expression(), rhs.expression()));
}

} // namespace zipper

#endif
