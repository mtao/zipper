#if !defined(ZIPPER_TENSORBASE_HPP)
#define ZIPPER_TENSORBASE_HPP

#include "ZipperBase.hpp"
#include "concepts/Tensor.hpp"
#include "concepts/stl.hpp"
#include "detail/extents/static_extents_to_integral_sequence.hpp"
#include "expression/nullary/StlMDArray.hpp"

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

  template <typename... Args>
    requires(!(concepts::Tensor<Args> && ...))
  TensorBase(Args &&...args)
      : Base(std::in_place, std::forward<Args>(args)...) {}

  auto operator=(concepts::Tensor auto const &v) -> TensorBase & {
    expression() = v.expression();
    return *this;
  }

  template <concepts::Tensor Other>
  TensorBase(const Other &other)
    requires(expression::concepts::WritableExpression<expression_type>)
      : TensorBase(other.expression()) {}
  template <concepts::Tensor Other>
  auto operator=(const Other &other) -> TensorBase &
    requires(expression::concepts::WritableExpression<expression_type>)
  {
    return operator=(other.expression());
  }
  auto operator=(concepts::Tensor auto &&v) -> TensorBase & {
    return Base::operator=(v.expression());
  }

  // Slice methods - construct wrapper in-place to avoid moving non-movable expressions
  template <typename... Slices, typename Self> auto slice(this Self&& self, Slices &&...slices) {
    using child_t = detail::member_child_storage_t<Self, expression_type>;
    using V = expression::unary::Slice<child_t,
                  detail::slice_type_for_t<std::decay_t<Slices>>...>;
    return TensorBase<V>(std::in_place, std::forward<Self>(self).expression(),
        Base::filter_args_for_zipperbase(std::forward<Slices>(slices))...);
  }
  template <typename... Slices, typename Self> auto slice(this Self&& self) {
    using child_t = detail::member_child_storage_t<Self, expression_type>;
    using V = expression::unary::Slice<child_t, std::decay_t<Slices>...>;
    return TensorBase<V>(std::in_place, std::forward<Self>(self).expression(), Slices{}...);
  }
};

template <concepts::Expression Expr>
TensorBase(Expr &&) -> TensorBase<Expr>;
template <concepts::Expression Expr>
TensorBase(const Expr &) -> TensorBase<Expr>;

// STL deduction guides: rvalue → owning StlMDArray, lvalue → borrowing StlMDArray
template <concepts::StlStorage S>
TensorBase(S &&) -> TensorBase<expression::nullary::StlMDArray<std::decay_t<S>>>;
template <concepts::StlStorage S>
TensorBase(S &) -> TensorBase<expression::nullary::StlMDArray<S &>>;

} // namespace zipper

#endif
