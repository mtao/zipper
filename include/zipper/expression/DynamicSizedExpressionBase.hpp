
#if !defined(ZIPPER_EXPRESSION_DYNAMICSIZEDEXPRESSIONBASE_HPP)
#define ZIPPER_EXPRESSION_DYNAMICSIZEDEXPRESSIONBASE_HPP

#include "ExpressionBase.hpp"
namespace zipper::expression {

/// An Expression that stores its own size
template <typename Derived_>
class DynamicSizedExpressionBase : public ExpressionBase<Derived_> {
public:
  using Derived = Derived_;
  using traits = detail::ExpressionTraits<Derived>;

  using Base = ExpressionBase<Derived>;
  using value_type = traits::value_type;
  using extents_type = traits::extents_type;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
  constexpr static bool IsStatic = extents_traits::is_static;
  auto derived() -> Derived & { return static_cast<Derived &>(*this); }
  auto derived() const -> const Derived & {
    return static_cast<const Derived &>(*this);
  }
  constexpr DynamicSizedExpressionBase()
    requires(IsStatic)
  {}

  constexpr auto extents() const -> extents_type {
    using ET = extents_traits::dynamic_indices_helper;
    auto run = [this]<rank_type... N>(
                   std::integer_sequence<rank_type, N...>) -> extents_type {
      return extents_type{extent(std::get<ET::dynamic_local_indnices>(N))...};
    };

    return run();
  }
};
} // namespace zipper::expression
#endif
