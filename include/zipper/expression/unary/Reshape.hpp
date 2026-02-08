#if !defined(ZIPPER_EXPRESSION_UNARY_RESHAPE_HPP)
#define ZIPPER_EXPRESSION_UNARY_RESHAPE_HPP

#include "UnaryExpressionBase.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/storage/layout_types.hpp"

namespace zipper::expression {
namespace unary {

template <zipper::concepts::QualifiedExpression ExpressionType,
          zipper::concepts::Extents NewExtents>
class Reshape;

namespace _detail_reshape {

/// Unravel a linear (flat) index into multi-dimensional indices for the
/// given extents, using layout_right (row-major) ordering.
template <zipper::concepts::Extents E>
constexpr auto unravel_index(index_type linear, const E &extents)
    -> std::array<index_type, E::rank()> {
  std::array<index_type, E::rank()> indices{};
  if constexpr (E::rank() > 0) {
    for (rank_type i = E::rank(); i-- > 0;) {
      indices[i] = linear % extents.extent(i);
      linear /= extents.extent(i);
    }
  }
  return indices;
}

} // namespace _detail_reshape

} // namespace unary

// ── Traits specialization ──────────────────────────────────────────────
template <zipper::concepts::QualifiedExpression ExpressionType,
          zipper::concepts::Extents NewExtents>
struct detail::ExpressionTraits<unary::Reshape<ExpressionType, NewExtents>>
    : public unary::detail::DefaultUnaryExpressionTraits<ExpressionType> {

  using child_traits = detail::ExpressionTraits<ExpressionType>;
  using value_type = typename child_traits::value_type;
  using extents_type = NewExtents;

  constexpr static bool is_coefficient_consistent = false;
  constexpr static bool is_value_based = false;

  constexpr static zipper::detail::AccessFeatures access_features =
      child_traits::access_features;
  consteval static auto is_const_valued() -> bool {
    return access_features.is_const;
  }
  consteval static auto is_reference_valued() -> bool {
    return access_features.is_reference;
  }
  consteval static auto is_assignable() -> bool {
    return !is_const_valued() && is_reference_valued();
  }
  consteval static auto is_referrable() -> bool {
    return access_features.is_reference;
  }
  constexpr static bool is_writable = is_assignable();
};

// ── Class definition ───────────────────────────────────────────────────
namespace unary {

template <zipper::concepts::QualifiedExpression ExpressionType,
          zipper::concepts::Extents NewExtents>
class Reshape : public UnaryExpressionBase<Reshape<ExpressionType, NewExtents>,
                                           ExpressionType> {
public:
  using self_type = Reshape<ExpressionType, NewExtents>;
  using traits = zipper::expression::detail::ExpressionTraits<self_type>;
  using extents_type = typename traits::extents_type;
  using value_type = typename traits::value_type;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

  using Base = UnaryExpressionBase<self_type, ExpressionType>;
  using Base::expression;

  using child_extents_type =
      typename zipper::expression::detail::ExpressionTraits<
          ExpressionType>::extents_type;
  using child_extents_traits =
      zipper::detail::ExtentsTraits<child_extents_type>;

  using layout_policy = zipper::default_layout_policy;
  using new_mapping_type =
      typename layout_policy::template mapping<extents_type>;

  Reshape() = delete;
  Reshape(const Reshape &) = default;
  Reshape(Reshape &&) = default;
  auto operator=(const Reshape &) -> Reshape & = delete;
  auto operator=(Reshape &&) -> Reshape & = delete;

  /// Construct from expression + new extents.
  /// Asserts total element count is preserved.
  Reshape(ExpressionType &expr, const extents_type &new_extents)
      : Base(expr), m_extents(new_extents), m_new_mapping(new_extents) {
    assert(extents_traits::size(new_extents) ==
           child_extents_traits::size(expr.extents()));
  }

  /// Construct with static new extents (when fully static).
  Reshape(ExpressionType &expr)
    requires(extents_traits::is_static)
      : Reshape(expr, extents_type{}) {}

  [[nodiscard]] constexpr auto extent(rank_type i) const -> index_type {
    return m_extents.extent(i);
  }

  constexpr auto extents() const -> extents_type { return m_extents; }

  /// Coefficient access: map new multi-index → linear → child multi-index
  template <typename... Args>
  auto coeff(Args &&...indices) const -> value_type {
    index_type linear = m_new_mapping(static_cast<index_type>(indices)...);
    return coeff_from_linear(
        linear,
        std::make_integer_sequence<rank_type, child_extents_type::rank()>{});
  }

private:
  template <rank_type... Ranks>
  auto coeff_from_linear(index_type linear,
                         std::integer_sequence<rank_type, Ranks...>) const
      -> value_type {
    auto child_indices =
        _detail_reshape::unravel_index(linear, expression().extents());
    return expression().coeff(child_indices[Ranks]...);
  }

  extents_type m_extents;
  new_mapping_type m_new_mapping;
};

template <zipper::concepts::QualifiedExpression E,
          zipper::concepts::Extents Ext>
Reshape(E &, const Ext &) -> Reshape<E, Ext>;

} // namespace unary
} // namespace zipper::expression
#endif
