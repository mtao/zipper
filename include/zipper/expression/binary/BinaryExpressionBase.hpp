#if !defined(ZIPPER_EXPRESSION_BINARY_BINARYEXPRESSIONBASE_HPP)
#define ZIPPER_EXPRESSION_BINARY_BINARYEXPRESSIONBASE_HPP

#include "zipper/concepts/Expression.hpp"
#include "zipper/expression/ExpressionBase.hpp"

namespace zipper::expression::binary {

namespace detail {

/// Default traits for binary expressions. Inherits from BasicExpressionTraits
/// to provide the access_features/shape_features interface.
template <zipper::concepts::QualifiedExpression ChildA,
          zipper::concepts::QualifiedExpression ChildB>
struct DefaultBinaryExpressionTraits
    : public expression::detail::BasicExpressionTraits<
          typename expression::detail::ExpressionTraits<ChildA>::value_type,
          zipper::dextents<0>,
          expression::detail::AccessFeatures{
              .is_const = true, .is_reference = false, .is_alias_free = true},
          expression::detail::ShapeFeatures{.is_resizable = false}> {
  using ATraits = expression::detail::ExpressionTraits<ChildA>;
  using BTraits = expression::detail::ExpressionTraits<ChildB>;
  using lhs_value_type = typename ATraits::value_type;
  using rhs_value_type = typename BTraits::value_type;
  static_assert(std::is_convertible_v<typename ATraits::value_type,
                                      typename BTraits::value_type> ||
                std::is_convertible_v<typename BTraits::value_type,
                                      typename ATraits::value_type>);

  // defaulting to first parameter
  using value_type = typename ATraits::value_type;
  constexpr static bool is_coefficient_consistent =
      expression::detail::get_is_coefficient_consistent<ATraits>() && expression::detail::get_is_coefficient_consistent<BTraits>();
  constexpr static bool is_value_based = true;
};
} // namespace detail

template <typename Derived, zipper::concepts::QualifiedExpression ChildTypeA,
          concepts::QualifiedExpression ChildTypeB>
class BinaryExpressionBase
    : public expression::ExpressionBase<Derived> {
public:
  using self_type = BinaryExpressionBase<Derived, ChildTypeA, ChildTypeB>;
  using traits = zipper::expression::detail::ExpressionTraits<Derived>;
  using extents_type = typename traits::extents_type;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
  using value_type = typename traits::value_type;
  constexpr static bool is_static = extents_traits::is_static;
  constexpr static bool is_value_based = traits::is_value_based;

  using Base = expression::ExpressionBase<Derived>;

  auto derived() -> Derived & { return static_cast<Derived &>(*this); }
  auto derived() const -> const Derived & {
    return static_cast<const Derived &>(*this);
  }
  using lhs_value_type = typename traits::lhs_value_type;
  using rhs_value_type = typename traits::rhs_value_type;

  BinaryExpressionBase(const BinaryExpressionBase &) = default;
  BinaryExpressionBase(BinaryExpressionBase &&) = default;
  auto operator=(const BinaryExpressionBase &)
      -> BinaryExpressionBase & = default;
  auto operator=(BinaryExpressionBase &&) -> BinaryExpressionBase & = default;

  /// storage helpers: use pointers for reference child types
  /// to preserve assignability (raw references delete copy/move assignment).
  using lhs_element_type = std::remove_reference_t<ChildTypeA>;
  using rhs_element_type = std::remove_reference_t<ChildTypeB>;
  using lhs_storage_type = std::conditional_t<
      std::is_lvalue_reference_v<ChildTypeA>,
      const lhs_element_type*,
      ChildTypeA>;
  using rhs_storage_type = std::conditional_t<
      std::is_lvalue_reference_v<ChildTypeB>,
      const rhs_element_type*,
      ChildTypeB>;

  BinaryExpressionBase(const lhs_element_type &a,
                       const rhs_element_type &b)
      : m_lhs(_init_lhs(a)), m_rhs(_init_rhs(b)) {}

  /// Default extents() constructs from extent() calls.
  /// Derived classes must provide extent(rank_type) — there is no single
  /// child to delegate to.
  constexpr auto extents() const -> extents_type {
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
    return extents_traits::make_extents_from(derived());
  }

  auto lhs() const -> const lhs_element_type & { return _get_lhs(); }

  auto rhs() const -> const rhs_element_type & { return _get_rhs(); }

  auto get_value(const lhs_value_type &l, const rhs_value_type &r) const
      -> value_type {
    return derived().get_value(l, r);
  }

  template <typename... Args>
  auto coeff(Args &&...args) const -> value_type
    requires(is_value_based)
  {
    return value_type(get_value(_get_lhs()(std::forward<Args>(args)...),
                                _get_rhs()(std::forward<Args>(args)...)));
  }

private:
  static auto _init_lhs(const lhs_element_type &a) -> lhs_storage_type {
    if constexpr (std::is_lvalue_reference_v<ChildTypeA>) {
      return std::addressof(a);
    } else {
      return a;
    }
  }
  static auto _init_rhs(const rhs_element_type &b) -> rhs_storage_type {
    if constexpr (std::is_lvalue_reference_v<ChildTypeB>) {
      return std::addressof(b);
    } else {
      return b;
    }
  }
  auto _get_lhs() const -> const lhs_element_type & {
    if constexpr (std::is_lvalue_reference_v<ChildTypeA>) {
      return *m_lhs;
    } else {
      return m_lhs;
    }
  }
  auto _get_rhs() const -> const rhs_element_type & {
    if constexpr (std::is_lvalue_reference_v<ChildTypeB>) {
      return *m_rhs;
    } else {
      return m_rhs;
    }
  }

  /// When ChildTypeA/B are reference types (e.g. `const Expr&`), the expression
  /// node stores pointers (non-owning, but assignable). When they are value
  /// types (e.g. `const Expr`), the node stores by value (owns copies).
  lhs_storage_type m_lhs;
  rhs_storage_type m_rhs;
};

} // namespace zipper::expression::binary
#endif
