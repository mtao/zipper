#if !defined(ZIPPER_expression_BINARY_BINARYVIEW_HPP)
#define ZIPPER_expression_BINARY_BINARYVIEW_HPP

#include "zipper/concepts/Expression.hpp"
#include "zipper/expression/SizedExpressionBase.hpp"

namespace zipper::expression::binary {

namespace detail {
template <zipper::concepts::QualifiedExpression ChildA,
          zipper::concepts::QualifiedExpression ChildB,
          bool _holds_extents = true>
struct DefaultBinaryExpressionTraits
    : public expression::detail::DefaultExpressionTraits<> {
  using ATraits = expression::detail::ExpressionTraits<ChildA>;
  using BTraits = expression::detail::ExpressionTraits<ChildB>;
  using lhs_value_type = ATraits::value_type;
  using rhs_value_type = BTraits::value_type;
  // using extents_type = typename BaseTraits::extents_type;
  static_assert(std::is_convertible_v<typename ATraits::value_type,
                                      typename BTraits::value_type> ||
                std::is_convertible_v<typename BTraits::value_type,
                                      typename ATraits::value_type>);

  // defaulting to first parameter
  using value_type = typename ATraits::value_type;
  constexpr static bool holds_extents = _holds_extents;
  constexpr static bool is_coefficient_consistent =
      ATraits::is_coefficient_consistent && BTraits::is_coefficient_consistent;
  constexpr static bool is_value_based = true;

  // to pass a base type to the BinaryExpressionBase
  template <typename Derived>
  using base_type =
      std::conditional_t<holds_extents, SizedExpressionBase<Derived>,
                         ExpressionBase<Derived>>;
};
} // namespace detail

template <typename Derived, zipper::concepts::QualifiedExpression ChildTypeA,
          concepts::QualifiedExpression ChildTypeB>
class BinaryExpressionBase : public expression::detail::ExpressionTraits<
                                 Derived>::template base_type<Derived> {
public:
  using Base = expression::detail::ExpressionTraits<
      Derived>::template base_type<Derived>;
  using self_type = BinaryExpressionBase<Derived, ChildTypeA, ChildTypeB>;
  using traits = zipper::expression::detail::ExpressionTraits<Derived>;
  using extents_type = traits::extents_type;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
  using value_type = traits::value_type;
  constexpr static bool holds_extents = traits::holds_extents;
  constexpr static bool is_static = extents_traits::is_static;
  constexpr static bool is_value_based = traits::is_value_based;
  auto derived() -> Derived & { return static_cast<Derived &>(*this); }
  auto derived() const -> const Derived & {
    return static_cast<const Derived &>(*this);
  }
  using lhs_value_type = traits::lhs_value_type;
  using rhs_value_type = traits::rhs_value_type;

  BinaryExpressionBase(const BinaryExpressionBase &) = default;
  BinaryExpressionBase(BinaryExpressionBase &&) = default;
  auto operator=(const BinaryExpressionBase &)
      -> BinaryExpressionBase & = delete;
  auto operator=(BinaryExpressionBase &&) -> BinaryExpressionBase & = delete;
  BinaryExpressionBase(const ChildTypeA &a, const ChildTypeB &b)
    requires(!holds_extents || is_static)
      : m_lhs(a), m_rhs(b) {}

  BinaryExpressionBase(const ChildTypeA &a, const ChildTypeB &b,
                       const extents_type &e)
    requires(holds_extents)
      : Base(e), m_lhs(a), m_rhs(b) {}
  using Base::extent;

  auto lhs() const -> const ChildTypeA & { return m_lhs; }

  auto rhs() const -> const ChildTypeB & { return m_rhs; }

  auto get_value(const lhs_value_type &l, const rhs_value_type &r) const
      -> value_type {
    return derived().get_value(l, r);
  }

  constexpr auto extents() const -> const extents_type &
    requires(holds_extents)
  {
    return Base::extents();
  }

  template <typename... Args>
  auto coeff(Args &&...args) const -> value_type
    requires(is_value_based)
  {
    return value_type(get_value(m_lhs(std::forward<Args>(args)...),
                                m_rhs(std::forward<Args>(args)...)));
  }

private:
  const ChildTypeA &m_lhs;
  const ChildTypeB &m_rhs;
};

} // namespace zipper::expression::binary
#endif
