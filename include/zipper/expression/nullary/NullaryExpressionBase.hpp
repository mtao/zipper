#if !defined(ZIPPER_EXPRESSION_NULLARY_NULLARYEXPRESSIONBASE_HPP)
#define ZIPPER_EXPRESSION_NULLARY_NULLARYEXPRESSIONBASE_HPP

#include "zipper/types.hpp"

namespace zipper::expression::nullary {

template <typename Derived, typename T, index_type... Indices>
class NullaryExpressionBase
    : public expression::detail::template ExpressionTraits<Derived>::base_type {
public:
  using self_type = NullaryExpressionBase<Derived, T, Indices...>;
  using traits = zipper::expression::detail::ExpressionTraits<Derived>;
  using extents_type = traits::extents_type;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
  using value_type = traits::value_type;
  constexpr static bool is_static = extents_traits::is_static;
  constexpr static bool is_coefficient_consistent =
      traits::is_coefficient_consistent;
  constexpr static bool is_value_based = traits::is_value_based;
  using Base = traits ::base_type;
  using Base::extent;

  auto derived() -> Derived & { return static_cast<Derived &>(*this); }
  auto derived() const -> const Derived & {
    return static_cast<const Derived &>(*this);
  }

  constexpr NullaryExpressionBase(const NullaryExpressionBase &) = default;
  constexpr NullaryExpressionBase(NullaryExpressionBase &&) = default;
  constexpr NullaryExpressionBase()
    requires(is_static)
  = default;

  constexpr NullaryExpressionBase(const extents_type &e) : Base(e) {}

  constexpr auto get_value() const -> value_type
    requires(is_value_based)
  {
    return derived().get_value();
  }

  template <typename... Args>
  constexpr auto coeff(Args &&...) const -> value_type
    requires(is_value_based)
  {
    return get_value();
  }
};

} // namespace zipper::expression::nullary
#endif
