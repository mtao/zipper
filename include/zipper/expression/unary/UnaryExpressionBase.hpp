#if !defined(ZIPPER_EXPRESSION_UNARY_UNARY_HPP)
#define ZIPPER_EXPRESSION_UNARY_UNARY_HPP

#include "zipper/concepts/Expression.hpp"
#include "zipper/expression/ExpressionBase.hpp"
#include "zipper/expression/SizedExpressionBase.hpp"

namespace zipper::expression::unary {

namespace detail {
template <zipper::concepts::Expression Child, bool _holds_extents = false>
struct DefaultUnaryExpressionTraits
    : public expression::detail::DefaultExpressionTraits<
          typename expression::detail::ExpressionTraits<Child>::value_type,
          typename expression::detail::ExpressionTraits<Child>::extents_type> {
  // to pass a base type to the UnaryExpressionBase
  constexpr static bool holds_extents = _holds_extents;
  constexpr static bool is_const = std::is_const_v<Child>;
  template <typename Derived>
  using base_type =
      std::conditional_t<holds_extents, SizedExpressionBase<Derived>,
                         ExpressionBase<Derived>>;
  using base_traits = expression::detail::ExpressionTraits<Child>;
  using base_value_type = base_traits::value_type;
  constexpr static bool is_coefficient_consistent =
      base_traits::is_coefficient_consistent;
  constexpr static bool is_value_based = true;
  constexpr static bool is_writable = base_traits::is_writable && !is_const;

  constexpr static bool is_rvalue_reference = std::is_rvalue_reference_v<Child>;
  constexpr static bool is_lvalue_reference = std::is_lvalue_reference_v<Child>;
  using child_type_ = std::decay_t<Child>;
  using child_type_refed =
      std::conditional<is_lvalue_reference, child_type_ &, child_type_>;
  using child_type =
      std::conditional_t<is_writable, child_type_refed, const child_type_refed>;
};
} // namespace detail

template <typename Derived, zipper::concepts::Expression ChildType>
class UnaryExpressionBase : public expression::detail::ExpressionTraits<
                                Derived>::template base_type<Derived> {
public:
  using self_type = UnaryExpressionBase<Derived, ChildType>;
  using traits = zipper::expression::detail::ExpressionTraits<Derived>;
  using extents_type = traits::extents_type;
  using value_type = traits::value_type;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
  constexpr static bool holds_extents = traits::holds_extents;
  constexpr static bool is_static = extents_traits::is_static;

  using Base = typename traits::template base_type<Derived>;
  // using Base =
  //     expression::detail::ExpressionTraits<Derived>::template
  //     base_type<Derived>;
  using Base::extent;
  constexpr static bool is_value_based = traits::is_value_based;
  constexpr static bool is_const = traits::is_const;
  constexpr static bool is_writable = traits::is_writable;
  using child_type =
      std::conditional_t<is_writable, ChildType, const ChildType>;
  // using child_type = typename traits::child_type;

  auto derived() -> Derived & { return static_cast<Derived &>(*this); }
  auto derived() const -> const Derived & {
    return static_cast<const Derived &>(*this);
  }
  using child_value_type = traits::base_value_type;
  UnaryExpressionBase() = delete;

  UnaryExpressionBase(const UnaryExpressionBase &) = default;
  UnaryExpressionBase(UnaryExpressionBase &&) = default;
  auto operator=(const UnaryExpressionBase &) -> UnaryExpressionBase & = delete;
  auto operator=(UnaryExpressionBase &&) -> UnaryExpressionBase & = delete;
  UnaryExpressionBase(child_type &b)
    requires(!holds_extents || is_static)
      : m_expression(b) {}
  UnaryExpressionBase(child_type &b, const extents_type &e)
      : Base(e), m_expression(b) {}

  constexpr auto extents() const -> const extents_type & {
    if constexpr (holds_extents) {
      return Base::extents();
    } else {
      return m_expression.extents();
    }
  }

  auto get_value(const child_value_type &value) const -> decltype(auto)
    requires(is_value_based)
  {
    return derived().get_value(value);
  }

  auto expression() -> child_type & { return m_expression; }
  auto expression() const -> const child_type & { return m_expression; }
  template <typename... Args>
  auto coeff(Args &&...args) const -> value_type
    requires(is_value_based)
  {
    return get_value(m_expression(std::forward<Args>(args)...));
  }

private:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"
  child_type &m_expression;
#pragma GCC diagnostic pop
};

} // namespace zipper::expression::unary
#endif
