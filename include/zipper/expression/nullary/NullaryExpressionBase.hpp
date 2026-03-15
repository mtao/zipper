#if !defined(ZIPPER_EXPRESSION_NULLARY_NULLARYEXPRESSIONBASE_HPP)
#define ZIPPER_EXPRESSION_NULLARY_NULLARYEXPRESSIONBASE_HPP

#include "zipper/expression/ExpressionBase.hpp"
#include "zipper/types.hpp"

namespace zipper::expression::nullary {

/// Helper class for nullary expressions to wrap a little get_value() entrypoint
template <typename Derived>
class NullaryExpressionBase : public ExpressionBase<Derived> {
public:
  using expression_base = ExpressionBase<Derived>;
  using self_type = NullaryExpressionBase<Derived>;
  using traits = zipper::expression::detail::ExpressionTraits<Derived>;
  using extents_type = traits::extents_type;
  using extents_traits = traits::extents_traits;
  using value_type = traits::value_type;
  constexpr static bool is_static = extents_traits::is_static;
  constexpr static expression::detail::AccessFeatures access_features =
      expression_base::access_features;

  auto derived(this auto& self) -> auto& {
    if constexpr (std::is_const_v<std::remove_reference_t<decltype(self)>>) {
      return static_cast<const Derived &>(self);
    } else {
      return static_cast<Derived &>(self);
    }
  }

  static_assert(!zipper::concepts::Extents<value_type>);
  constexpr auto get_value() const -> value_type {
    value_type v = derived().get_value();
    return v;
  }

  template <zipper::concepts::Index... Args>
  constexpr auto coeff(Args &&...) const -> value_type {
    return get_value();
  }
};

} // namespace zipper::expression::nullary
#endif
