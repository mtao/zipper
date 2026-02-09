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

namespace _ueb_detail {
/// Helper: adds const through lvalue references.
/// - `T`         → `const T`
/// - `T&`        → `const T&`
/// - `const T`   → `const T`
/// - `const T&`  → `const T&`
template <typename T>
using add_const_through_ref_t = std::conditional_t<
    std::is_lvalue_reference_v<T>,
    const std::remove_reference_t<T>&,
    const T
>;
} // namespace _ueb_detail

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
  /// child_type: when ChildType is a reference (e.g. `Expr&` or `const Expr&`),
  /// the expression node stores a reference (non-owning view).
  /// When ChildType is a value (e.g. `Expr` or `const Expr`),
  /// the expression node stores by value (owns a copy).
  using child_type =
      std::conditional_t<is_assignable, ChildType,
                         _ueb_detail::add_const_through_ref_t<ChildType>>;

  /// storage_type: when child_type is a reference, we use a pointer
  /// to preserve assignability (raw references delete copy/move assignment).
  /// When child_type is a value type, we store directly.
  using child_element_type = std::remove_reference_t<child_type>;
  using storage_type = std::conditional_t<
      std::is_lvalue_reference_v<child_type>,
      child_element_type*,
      child_type>;
  /// constructor_arg_type: the reference type that derived constructors
  /// should use when accepting the child expression.
  /// - For value types or const ref: `const child_element_type &`
  /// - For non-const lvalue ref (mutable view): `child_element_type &`
  using constructor_arg_type = std::conditional_t<
      std::is_lvalue_reference_v<child_type> &&
          !std::is_const_v<child_element_type>,
      child_element_type &,
      const child_element_type &>;

  auto derived() -> Derived & { return static_cast<Derived &>(*this); }
  auto derived() const -> const Derived & {
    return static_cast<const Derived &>(*this);
  }
  using child_value_type = typename traits::base_value_type;
  UnaryExpressionBase() = delete;

  UnaryExpressionBase(const UnaryExpressionBase &) = default;
  UnaryExpressionBase(UnaryExpressionBase &&) = default;
  auto operator=(const UnaryExpressionBase &) -> UnaryExpressionBase & = default;
  auto operator=(UnaryExpressionBase &&) -> UnaryExpressionBase & = default;

  UnaryExpressionBase(constructor_arg_type b)
      : m_expression(_init_storage(b)) {}

  /// Primary extent accessor — delegates directly to child expression.
  /// Derived classes that change shape should shadow this.
  constexpr auto extent(rank_type i) const -> index_type {
    return _get_expression().extent(i);
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

  auto expression() -> child_element_type & { return _get_expression(); }
  auto expression() const -> const child_element_type & { return _get_expression(); }
  template <typename... Args>
  auto coeff(Args &&...args) const -> value_type
    requires(is_value_based)
  {
    return get_value(_get_expression()(std::forward<Args>(args)...));
  }

private:
  static auto _init_storage(constructor_arg_type b) -> storage_type {
    if constexpr (std::is_lvalue_reference_v<child_type>) {
      return std::addressof(b);
    } else {
      return b;
    }
  }
  auto _get_expression() -> child_element_type & {
    if constexpr (std::is_lvalue_reference_v<child_type>) {
      return *m_expression;
    } else {
      return m_expression;
    }
  }
  auto _get_expression() const -> const child_element_type & {
    if constexpr (std::is_lvalue_reference_v<child_type>) {
      return *m_expression;
    } else {
      return m_expression;
    }
  }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"
  storage_type m_expression;
#pragma GCC diagnostic pop
};

} // namespace zipper::expression::unary
#endif
