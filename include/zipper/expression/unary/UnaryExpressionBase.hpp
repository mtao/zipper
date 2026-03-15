#if !defined(ZIPPER_EXPRESSION_UNARY_UNARYEXPRESSIONBASE_HPP)
#define ZIPPER_EXPRESSION_UNARY_UNARYEXPRESSIONBASE_HPP

#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/ExpressionStorage.hpp"
#include "zipper/expression/ExpressionBase.hpp"

namespace zipper::expression::unary {

namespace detail {

/// Helper to compute access_features that accounts for const qualification
/// of the child expression. When Child is const-qualified, is_const must
/// be true regardless of what the child's own traits report.
namespace _dut_detail {
template <zipper::concepts::QualifiedExpression Child>
consteval auto const_corrected_access_features()
    -> zipper::detail::AccessFeatures {
  constexpr auto base =
      expression::detail::ExpressionTraits<std::decay_t<Child>>::access_features;
  return {
      .is_const = base.is_const ||
                  std::is_const_v<std::remove_reference_t<Child>>,
      .is_reference = base.is_reference,
  };
}
} // namespace _dut_detail

/// Default implementation details for unary expressions.
///
/// Holds child traits alias and child value type that the unary expression
/// class body needs for its implementation. These are NOT part of the
/// public traits interface (capability flags, value_type, extents_type).
///
/// Child is the *qualified* child type as passed to the unary expression
/// class template (e.g. `const MDArray<…>&` for reference storage,
/// `const MDArray<…>` for owning storage).  Traits are looked up on the
/// decayed (unqualified) type.
template <zipper::concepts::QualifiedExpression Child>
struct DefaultUnaryExpressionDetail {
  using child_traits = expression::detail::ExpressionTraits<std::decay_t<Child>>;
  using base_value_type = typename child_traits::value_type;
};

/// Default traits for unary expressions. Inherits from BasicExpressionTraits
/// to provide the access_features/shape_features interface.
///
/// Child is the *qualified* child type as passed to the unary expression
/// class template (e.g. `const MDArray<…>&` for reference storage,
/// `const MDArray<…>` for owning storage).  Traits are looked up on the
/// decayed (unqualified) type.
///
/// AF controls the access features (writability, referrability) for this
/// expression.  Defaults to the child's access features with
/// const-correction applied (i.e. if Child is const-qualified, is_const
/// is forced to true).  Override this parameter for expressions that
/// compute values on the fly and cannot return references (e.g.
/// CoefficientWiseOperation, ScalarOperation) — typically with
/// `{.is_const = true, .is_reference = false}`.
template <zipper::concepts::QualifiedExpression Child,
          zipper::detail::AccessFeatures AF =
              _dut_detail::const_corrected_access_features<Child>()>
struct DefaultUnaryExpressionTraits
    : public expression::detail::BasicExpressionTraits<
          typename expression::detail::ExpressionTraits<std::decay_t<Child>>::value_type,
          typename expression::detail::ExpressionTraits<std::decay_t<Child>>::extents_type,
          AF,
          expression::detail::ShapeFeatures{.is_resizable = false}> {
  using _Detail = DefaultUnaryExpressionDetail<Child>;
  using base_value_type = typename _Detail::base_value_type;

  constexpr static bool is_value_based = true;
  constexpr static bool is_coefficient_consistent =
      _Detail::child_traits::is_coefficient_consistent;

  /// stores_references is true when the child is stored by reference,
  /// OR when the child itself (even if stored by value) internally stores
  /// references to external data.  This ensures that an expression like
  /// -(a+b) correctly reports stores_references==true when (a+b) is moved
  /// in by value but still holds references to a and b.
  constexpr static bool stores_references =
      std::is_reference_v<Child> || _Detail::child_traits::stores_references;
};
} // namespace detail

/// Base class for unary expression nodes.
///
/// ChildType should be ref-qualified (e.g. `ExprType&` or `const ExprType&`)
/// when the caller wants reference storage (the common case — the child
/// outlives the expression node).  Pass a non-reference type to trigger
/// by-value storage (e.g. for temporaries that must be owned).
///
/// The storage mechanism is provided by expression_storage_t:
///   - `ExprType&`       →  stores as `ExprType&`       (mutable reference)
///   - `const ExprType&` →  stores as `const ExprType&` (const reference)
///   - `ExprType`        →  stores as `ExprType`        (by value, owns it)
template <typename Derived, typename ChildType>
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

  using storage_type = zipper::detail::expression_storage_t<ChildType>;
  using expression_type = std::remove_reference_t<ChildType>;
  // child_type preserves the const-qualification convention:
  // if not assignable, the child is viewed as const.
  using child_type =
      std::conditional_t<is_assignable, expression_type, const expression_type>;

  auto derived(this auto& self) -> auto& {
    if constexpr (std::is_const_v<std::remove_reference_t<decltype(self)>>) {
      return static_cast<const Derived &>(self);
    } else {
      return static_cast<Derived &>(self);
    }
  }
  using child_value_type = typename traits::base_value_type;
  UnaryExpressionBase() = delete;

  UnaryExpressionBase(const UnaryExpressionBase &) = default;
  UnaryExpressionBase(UnaryExpressionBase &&) = default;
  auto operator=(const UnaryExpressionBase &) -> UnaryExpressionBase & = delete;
  auto operator=(UnaryExpressionBase &&) -> UnaryExpressionBase & = delete;

  template <typename U>
    requires std::constructible_from<storage_type, U &&>
  UnaryExpressionBase(U &&b)
      : m_expression(std::forward<U>(b)) {}

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
  storage_type m_expression;
#pragma GCC diagnostic pop
};

} // namespace zipper::expression::unary
#endif
