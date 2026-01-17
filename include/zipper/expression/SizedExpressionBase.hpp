#if !defined(ZIPPER_EXPRESSIONS_SIZEDEDEXPRESSIONBASE_HPP)
#define ZIPPER_EXPRESSIONS_SIZEDEDEXPRESSIONBASE_HPP

#include "ExpressionBase.hpp"
namespace zipper::expression {

/// An Expression that stores its own size
template <typename Derived_>
class SizedExpressionBase : public ExpressionBase<Derived_> {
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
  constexpr SizedExpressionBase()
    requires(IsStatic)
      : m_extents({}) {}
  template <typename... Args>
  constexpr SizedExpressionBase(const extents_type &extents)
      : m_extents(extents) {}

  constexpr auto extents() const -> const extents_type & { return m_extents; }

  [[nodiscard]] constexpr auto extent(rank_type i) const -> index_type {
    return m_extents.extent(i);
    return extents_type::static_extent(i);
  }

protected:
  void set_extent(const extents_type &e) { m_extents = e; }

private:
  extents_type m_extents;
};
} // namespace zipper::expression
#endif
