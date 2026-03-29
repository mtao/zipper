#if !defined(ZIPPER_EXPRESSION_UNARY_EXTENTVIEW_HPP)
#define ZIPPER_EXPRESSION_UNARY_EXTENTVIEW_HPP

#include "UnaryExpressionBase.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/assert.hpp"
#include "zipper/expression/detail/IndexSet.hpp"

namespace zipper::expression {
namespace unary {

/// ExtentView wraps an expression and re-types its extents without
/// changing the actual data layout, index mapping, or element access.
///
/// This enables zero-cost static↔dynamic extent conversions:
///   - as_dynamic(expr)          → erases all static extents to dextents<N>
///   - as_extents<Ext>(expr)     → casts to specific extents (debug-asserted)
///
/// The child expression's rank must equal the new extents' rank.
/// When converting from dynamic to static extents, ZIPPER_ASSERT checks
/// that runtime dimension sizes match (zero overhead in release builds).
template <zipper::concepts::QualifiedExpression ExpressionType,
          zipper::concepts::Extents NewExtents>
class ExtentView;

} // namespace unary

// ── ExpressionDetail specialization ──────────────────────────────────
template <zipper::concepts::QualifiedExpression ExpressionType,
          zipper::concepts::Extents NewExtents>
struct detail::ExpressionDetail<unary::ExtentView<ExpressionType, NewExtents>> {
  using child_traits = detail::ExpressionTraits<std::decay_t<ExpressionType>>;
};

// ── ExpressionTraits specialization ──────────────────────────────────
template <zipper::concepts::QualifiedExpression ExpressionType,
          zipper::concepts::Extents NewExtents>
struct detail::ExpressionTraits<unary::ExtentView<ExpressionType, NewExtents>>
    : public unary::detail::DefaultUnaryExpressionTraits<ExpressionType> {

  using child_traits = detail::ExpressionTraits<std::decay_t<ExpressionType>>;
  using value_type = typename child_traits::value_type;
  using extents_type = NewExtents;

  // ExtentView only re-types extents; it does not remap indices.
  // We set is_value_based = false because we need to forward coeff_ref
  // and const_coeff_ref (which is_value_based doesn't support).
  constexpr static bool is_value_based = false;

  // Coefficient consistency is inherited — no iteration order change.
  constexpr static bool is_coefficient_consistent =
      child_traits::is_coefficient_consistent;

  // Preserve sparse index set capability.
  constexpr static bool has_index_set = child_traits::has_index_set;
};

// ── Class definition ─────────────────────────────────────────────────
namespace unary {

template <zipper::concepts::QualifiedExpression ExpressionType,
          zipper::concepts::Extents NewExtents>
class ExtentView
    : public UnaryExpressionBase<ExtentView<ExpressionType, NewExtents>,
                                 ExpressionType> {
public:
  using self_type = ExtentView<ExpressionType, NewExtents>;
  using traits = zipper::expression::detail::ExpressionTraits<self_type>;
  using extents_type = typename traits::extents_type;
  using value_type = typename traits::value_type;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

  using Base = UnaryExpressionBase<self_type, ExpressionType>;
  using Base::expression;

  using child_extents_type =
      typename zipper::expression::detail::ExpressionTraits<
          std::decay_t<ExpressionType>>::extents_type;

  // Rank must be preserved.
  static_assert(extents_type::rank() == child_extents_type::rank(),
                "ExtentView: new extents must have the same rank as child");

  ExtentView() = delete;
  ExtentView(const ExtentView &) = default;
  ExtentView(ExtentView &&) = default;
  auto operator=(const ExtentView &) -> ExtentView & = delete;
  auto operator=(ExtentView &&) -> ExtentView & = delete;

  /// Construct from expression.
  /// When the new extents have static dimensions, ZIPPER_ASSERT verifies
  /// that each static dimension matches the child's runtime extent.
  template <typename U>
    requires std::constructible_from<typename Base::storage_type, U &&>
  explicit ExtentView(U &&expr) : Base(std::forward<U>(expr)) {
    // Verify static extents match runtime sizes
    verify_extents_(
        std::make_integer_sequence<rank_type, extents_type::rank()>{});
  }

  // ── Extent access ────────────────────────────────────────────────
  [[nodiscard]] constexpr auto extent(rank_type i) const -> index_type {
    // For static dimensions, return the static value.
    // For dynamic dimensions, delegate to child.
    // In practice, since the child has the same runtime sizes,
    // we can always delegate to child.
    return expression().extent(i);
  }

  constexpr auto extents() const -> extents_type {
    return extents_traits::make_extents_from(*this);
  }

  // ── Coefficient access (pure forwarding) ─────────────────────────
  template <typename... Args>
  auto coeff(Args &&...indices) const -> value_type {
    return expression().coeff(std::forward<Args>(indices)...);
  }

  template <typename... Args>
  auto const_coeff_ref(Args &&...indices) const -> const value_type &
    requires(traits::is_referrable())
  {
    return expression().const_coeff_ref(std::forward<Args>(indices)...);
  }

  template <typename... Args>
  auto coeff_ref(Args &&...indices) -> value_type &
    requires(traits::is_assignable())
  {
    return expression().coeff_ref(std::forward<Args>(indices)...);
  }

  // ── Index set forwarding (sparse support) ────────────────────────
  template <rank_type D, typename... Args>
    requires(traits::has_index_set)
  auto index_set(Args &&...args) const {
    return expression().template index_set<D>(
        std::forward<Args>(args)...);
  }

  template <rank_type D, typename... Args>
    requires(traits::has_index_set)
  auto nonzero_range(Args &&...args) const {
    return index_set<D>(std::forward<Args>(args)...);
  }

  auto col_range_for_row(index_type row) const
    requires(traits::has_index_set && extents_type::rank() == 2)
  {
    return expression().col_range_for_row(row);
  }

  auto row_range_for_col(index_type col) const
    requires(traits::has_index_set && extents_type::rank() == 2)
  {
    return expression().row_range_for_col(col);
  }

  auto nonzero_segment() const
    requires(traits::has_index_set && extents_type::rank() == 1)
  {
    return expression().nonzero_segment();
  }

  /// Recursively deep-copy child so the result owns all data.
  auto make_owned() const {
    auto owned_child = expression().make_owned();
    return ExtentView<const decltype(owned_child), NewExtents>(
        std::move(owned_child));
  }

private:
  /// Verify that each static extent in NewExtents matches the child's
  /// runtime extent. Only fires ZIPPER_ASSERT for dimensions that are
  /// statically specified in the new extents.
  template <rank_type... Ds>
  void verify_extents_(std::integer_sequence<rank_type, Ds...>) {
    (verify_one_extent_<Ds>(), ...);
  }

  template <rank_type D>
  void verify_one_extent_() {
    if constexpr (extents_type::static_extent(D) != dynamic_extent) {
      ZIPPER_ASSERT(expression().extent(D) ==
                    extents_type::static_extent(D));
    }
  }
};

} // namespace unary
} // namespace zipper::expression

// ══════════════════════════════════════════════════════════════════════
// Free functions — placed in namespace zipper for convenient use.
// ══════════════════════════════════════════════════════════════════════

namespace zipper {

/// Erase all static extents to fully dynamic extents.
///
/// Returns a lightweight view that wraps the expression with
/// dextents<N> (all dimensions dynamic). This is useful for passing
/// statically-typed expressions to functions that accept dynamic extents.
///
/// @example
///   Matrix<double, 3, 3> A;
///   auto dyn = as_dynamic(A.expression());  // extents = dextents<2>
template <concepts::Expression E>
auto as_dynamic(E &expr) {
  constexpr rank_type N = expression::detail::ExpressionTraits<E>::extents_type::rank();
  return expression::unary::ExtentView<E &, dextents<N>>(expr);
}

template <concepts::Expression E>
auto as_dynamic(const E &expr) {
  constexpr rank_type N = expression::detail::ExpressionTraits<E>::extents_type::rank();
  return expression::unary::ExtentView<const E &, dextents<N>>(expr);
}

/// Cast an expression's extents to a specific type.
///
/// When converting from dynamic to static extents, ZIPPER_ASSERT verifies
/// that the runtime dimension sizes match the static values (zero overhead
/// in release builds).
///
/// @tparam NewExtents  The target extents type (e.g. extents<3,3>)
///
/// @example
///   auto dyn_expr = ...;  // has dextents<2> with runtime sizes (3, 3)
///   auto stat = as_extents<extents<3, 3>>(dyn_expr);  // asserted in debug
template <concepts::Extents NewExtents, concepts::Expression E>
auto as_extents(E &expr) {
  return expression::unary::ExtentView<E &, NewExtents>(expr);
}

template <concepts::Extents NewExtents, concepts::Expression E>
auto as_extents(const E &expr) {
  return expression::unary::ExtentView<const E &, NewExtents>(expr);
}

} // namespace zipper

#endif
