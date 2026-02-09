#if !defined(ZIPPER_EXPRESSION_UNARY_UNARYEXPRESSIONBASE_HPP)
#define ZIPPER_EXPRESSION_UNARY_UNARYEXPRESSIONBASE_HPP

#include "zipper/concepts/Expression.hpp"
#include "zipper/expression/ExpressionBase.hpp"

namespace zipper::expression::unary {

namespace detail {

/// Helper to compute access_features that accounts for const qualification
/// of the child expression. When Child is const-qualified, is_const must
/// be true regardless of what the child's own traits report.
namespace _dut_detail {
template <concepts::QualifiedExpression Child>
consteval auto const_corrected_access_features()
    -> zipper::detail::AccessFeatures {
  constexpr auto base =
      expression::detail::ExpressionTraits<Child>::access_features;
  return {
      .is_const = base.is_const ||
                  std::is_const_v<std::remove_reference_t<Child>>,
      .is_reference = base.is_reference,
      .is_alias_free = base.is_alias_free,
  };
}
} // namespace _dut_detail

/// Default traits for unary expressions. Inherits from BasicExpressionTraits
/// to provide the access_features/shape_features interface.
template <zipper::concepts::Expression Child>
struct DefaultUnaryExpressionTraits
    : public expression::detail::BasicExpressionTraits<
          typename expression::detail::ExpressionTraits<Child>::value_type,
          typename expression::detail::ExpressionTraits<Child>::extents_type,
          _dut_detail::const_corrected_access_features<Child>(),
          expression::detail::ShapeFeatures{.is_resizable = false}> {
  using child_traits = expression::detail::ExpressionTraits<Child>;
  using base_value_type = typename child_traits::value_type;

  constexpr static bool is_value_based = true;
  constexpr static bool is_coefficient_consistent =
      expression::detail::get_is_coefficient_consistent<child_traits>();
};
} // namespace detail

template <typename Derived, zipper::concepts::Expression ChildType>
class UnaryExpressionBase
    : public expression::ExpressionBase<Derived> {
public:
  using self_type = UnaryExpressionBase<Derived, ChildType>;
  using traits = zipper::expression::detail::ExpressionTraits<Derived>;
  using extents_type = typename traits::extents_type;
  using value_type = typename traits::value_type;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
  constexpr static bool is_static = extents_traits::is_static;

  using Base = expression::ExpressionBase<Derived>;
  constexpr static bool is_value_based = traits::is_value_based;
  constexpr static bool is_const_valued = traits::access_features.is_const;
  constexpr static bool is_assignable = traits::is_assignable();
  using child_type =
      std::conditional_t<is_assignable, ChildType, const ChildType>;

  auto derived() -> Derived & { return static_cast<Derived &>(*this); }
  auto derived() const -> const Derived & {
    return static_cast<const Derived &>(*this);
  }
  using child_value_type = typename traits::base_value_type;
  UnaryExpressionBase() = delete;

  UnaryExpressionBase(const UnaryExpressionBase &) = default;
  UnaryExpressionBase(UnaryExpressionBase &&) = default;
  auto operator=(const UnaryExpressionBase &) -> UnaryExpressionBase & = delete;
  auto operator=(UnaryExpressionBase &&) -> UnaryExpressionBase & = delete;
  UnaryExpressionBase(child_type &b)
      : m_expression(b) {}

  /// Primary extent accessor — delegates directly to child expression.
  /// Derived classes that change shape should shadow this.
  constexpr auto extent(rank_type i) const -> index_type {
    return m_expression.extent(i);
  }

  /// Default extents() constructs from extent() calls.
  /// Derived classes that have different extents shadow this.
  constexpr auto extents() const -> extents_type {
    return extents_traits::make_extents_from(*this);
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
