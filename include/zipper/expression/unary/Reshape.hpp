#if !defined(ZIPPER_EXPRESSION_UNARY_RESHAPE_HPP)
#define ZIPPER_EXPRESSION_UNARY_RESHAPE_HPP

#include "UnaryExpressionBase.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/assert.hpp"
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

  using child_traits = detail::ExpressionTraits<std::decay_t<ExpressionType>>;
  using value_type = typename child_traits::value_type;
  using extents_type = NewExtents;

  constexpr static bool is_coefficient_consistent = false;
  constexpr static bool is_value_based = false;
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
          std::decay_t<ExpressionType>>::extents_type;
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
  template <typename U>
    requires std::constructible_from<typename Base::storage_type, U &&>
  Reshape(U &&expr, const extents_type &new_extents)
      : Base(std::forward<U>(expr)), m_extents(new_extents), m_new_mapping(new_extents) {
    ZIPPER_ASSERT(extents_traits::size(new_extents) ==
           child_extents_traits::size(expression().extents()));
  }

  /// Construct with static new extents (when fully static).
  template <typename U>
    requires (extents_traits::is_static && std::constructible_from<typename Base::storage_type, U &&>)
  Reshape(U &&expr)
      : Reshape(std::forward<U>(expr), extents_type{}) {}

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

  template <typename... Args>
  auto const_coeff_ref(Args &&...indices) const -> const value_type &
    requires(traits::is_referrable())
  {
    index_type linear = m_new_mapping(static_cast<index_type>(indices)...);
    return const_coeff_ref_from_linear(
        linear,
        std::make_integer_sequence<rank_type, child_extents_type::rank()>{});
  }

  template <typename... Args>
  auto coeff_ref(Args &&...indices) -> value_type &
    requires(traits::is_assignable())
  {
    index_type linear = m_new_mapping(static_cast<index_type>(indices)...);
    return coeff_ref_from_linear(
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

  template <rank_type... Ranks>
  auto const_coeff_ref_from_linear(
      index_type linear,
      std::integer_sequence<rank_type, Ranks...>) const -> const value_type & {
    auto child_indices =
        _detail_reshape::unravel_index(linear, expression().extents());
    return expression().const_coeff_ref(child_indices[Ranks]...);
  }

  template <rank_type... Ranks>
  auto coeff_ref_from_linear(index_type linear,
                             std::integer_sequence<rank_type, Ranks...>)
      -> value_type & {
    auto child_indices =
        _detail_reshape::unravel_index(linear, expression().extents());
    return expression().coeff_ref(child_indices[Ranks]...);
  }

  extents_type m_extents;
  new_mapping_type m_new_mapping;

public:
  /// Recursively deep-copy child so the result owns all data.
  auto make_owned() const {
      auto owned_child = expression().make_owned();
      return Reshape<const decltype(owned_child), NewExtents>(
          std::move(owned_child), m_extents);
  }
};

template <zipper::concepts::Expression E,
          zipper::concepts::Extents Ext>
Reshape(E &, const Ext &) -> Reshape<E&, Ext>;

template <zipper::concepts::Expression E,
          zipper::concepts::Extents Ext>
Reshape(const E &, const Ext &) -> Reshape<const E&, Ext>;

} // namespace unary
} // namespace zipper::expression
#endif
