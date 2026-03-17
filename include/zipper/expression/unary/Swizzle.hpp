#if !defined(ZIPPER_EXPRESSION_UNARY_SWIZZLE_HPP)
#define ZIPPER_EXPRESSION_UNARY_SWIZZLE_HPP

#include "UnaryExpressionBase.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/extents/swizzle_extents.hpp"
#include "zipper/expression/detail/AssignHelper.hpp"
#include "zipper/expression/detail/IndexSet.hpp"
#include "zipper/utils/extents/all_extents_indices.hpp"

namespace zipper::expression {
namespace unary {
template <zipper::concepts::QualifiedExpression ExpressionType,
          index_type... Indices>
class Swizzle;

} // namespace unary

/// Implementation details for Swizzle expressions.
///
/// Holds the decayed expression type alias and the swizzler utility type.
/// These are needed by the Swizzle class body for index remapping.
template <zipper::concepts::QualifiedExpression QualifiedExprType,
          index_type... Indices>
struct detail::ExpressionDetail<
    unary::Swizzle<QualifiedExprType, Indices...>> {
  using ExprType = std::decay_t<QualifiedExprType>;
  using swizzler_type = zipper::detail::extents::ExtentsSwizzler<Indices...>;
};

template <zipper::concepts::QualifiedExpression QualifiedExprType,
          index_type... Indices>
struct detail::ExpressionTraits<
    unary::Swizzle<QualifiedExprType, Indices...>>
    : public zipper::expression::unary::detail::DefaultUnaryExpressionTraits<
          QualifiedExprType> {
  using _Detail = detail::ExpressionDetail<unary::Swizzle<QualifiedExprType, Indices...>>;
  using Base = ExpressionTraits<std::decay_t<QualifiedExprType>>;
  using extents_type = typename _Detail::swizzler_type::template extents_type_swizzler_t<
      typename Base::extents_type>;
  using value_type = Base::value_type;
  constexpr static bool is_coefficient_consistent = false;
  constexpr static bool is_value_based = false;

  /// Propagate has_index_set from child — Swizzle permutes dimensions
  /// but does not change the sparsity structure.
  constexpr static bool has_index_set = Base::has_index_set;

  /// Backward-compatible alias for has_index_set.
  constexpr static bool has_known_zeros = has_index_set;
};

namespace unary {
template <zipper::concepts::QualifiedExpression QualifiedExprType,
          index_type... Indices>
class Swizzle
    : public UnaryExpressionBase<
          Swizzle<QualifiedExprType, Indices...>,
          QualifiedExprType> {
public:
  using self_type = Swizzle<QualifiedExprType, Indices...>;
  using traits = zipper::expression::detail::ExpressionTraits<self_type>;
  using detail_type = zipper::expression::detail::ExpressionDetail<self_type>;
  using ExprType = typename detail_type::ExprType;
  using extents_type = traits::extents_type;
  using value_type = traits::value_type;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
  using swizzler_type = typename detail_type::swizzler_type;
  using Base = UnaryExpressionBase<self_type, QualifiedExprType>;
  using Base::expression;
  constexpr static rank_type internal_rank = ExprType::extents_type::rank();
  constexpr static std::array<rank_type, internal_rank>
      to_internal_rank_indices = swizzler_type::valid_internal_indices;

  Swizzle(const Swizzle &) = default;
  Swizzle(Swizzle &&) = default;
  auto operator=(const Swizzle &) -> Swizzle & = delete;
  auto operator=(Swizzle &&) -> Swizzle & = delete;
  template <typename U>
    requires std::constructible_from<typename Base::storage_type, U &&>
  Swizzle(U &&b)
      : Base(std::forward<U>(b)) {}

  /// Recursively deep-copy child so the result owns all data.
  auto make_owned() const {
      auto owned_child = expression().make_owned();
      return Swizzle<const decltype(owned_child), Indices...>(
          std::move(owned_child));
  }

  // ── Index set propagation ───────────────────────────────────────────
  // For a Swizzle that permutes dimensions, index_set<D>(other_idx)
  // maps to index_set<child_dim>(other_idx) on the child, where
  // child_dim = swizzle_map[D].

  template <rank_type D>
    requires(traits::has_index_set && D < sizeof...(Indices))
  auto index_set(index_type other_idx) const {
    constexpr std::array<index_type, sizeof...(Indices)> swizzle_map = {{Indices...}};
    constexpr index_type child_dim = swizzle_map[D];
    // Inserted dimensions (dynamic_extent) have no child mapping;
    // they have extent 1 and are always "nonzero" at index 0.
    if constexpr (child_dim == std::dynamic_extent) {
      return zipper::expression::detail::SingleIndexRange{index_type{0}};
    } else {
      return expression().template index_set<child_dim>(other_idx);
    }
  }

  /// @deprecated Use index_set instead.
  template <rank_type D>
    requires(traits::has_index_set && D < sizeof...(Indices))
  auto nonzero_range(index_type other_idx) const {
    return index_set<D>(other_idx);
  }

  /// Convenience: col_range_for_row (rank-2 only).
  auto col_range_for_row(index_type row) const
    requires(traits::has_index_set && extents_type::rank() == 2)
  {
    return index_set<1>(row);
  }

  /// Convenience: row_range_for_col (rank-2 only).
  auto row_range_for_col(index_type col) const
    requires(traits::has_index_set && extents_type::rank() == 2)
  {
    return index_set<0>(col);
  }

  constexpr auto extent(rank_type i) const -> index_type {
      constexpr std::array<index_type, sizeof...(Indices)> swizzle_map = {{Indices...}};
      if (swizzle_map[i] == std::dynamic_extent) {
          return 1;
      } else {
          return expression().extent(swizzle_map[i]);
      }
  }

  constexpr auto extents() const -> extents_type {
    return extents_traits::make_extents_from(*this);
  }

  template <typename... Args>
    requires(extents_type::rank() == sizeof...(Args))
  value_type coeff(Args &&...idxs) const {
    return _coeff(swizzler_type::unswizzle(std::forward<Args>(idxs)...),
                  std::make_integer_sequence<rank_type, internal_rank>{});
  }
  template <typename... Args>
  value_type &coeff_ref(Args &&...idxs)
    requires((traits::is_assignable()) &&
             (extents_type::rank() == sizeof...(Args)))
  {
    return _coeff_ref(
        swizzler_type::unswizzle(std::forward<Args>(idxs)...),
        std::make_integer_sequence<rank_type, internal_rank>{});
  }
  template <typename... Args>
  const value_type &const_coeff_ref(Args &&...idxs) const
    requires((traits::is_referrable()) &&
             (extents_type::rank() == sizeof...(Args)))
  {
    return _const_coeff_ref(
        swizzler_type::unswizzle(std::forward<Args>(idxs)...),
        std::make_integer_sequence<rank_type, internal_rank>{});
  }

  template <zipper::concepts::Expression V>
  void assign(const V &v)
    requires(
        traits::is_assignable() &&
        extents_traits::template is_convertable_from<
            typename zipper::expression::detail::ExpressionTraits<
                V>::extents_type>())
  {
    expression::detail::AssignHelper<V, self_type>::assign(v, *this);
  }

private:
  template <zipper::concepts::IndexPackTuple T, rank_type... ranks>
  auto _coeff(const T &idxs,
              std::integer_sequence<rank_type, ranks...>) const -> value_type {
    return expression().coeff(std::get<ranks>(idxs)...);
  }
  template <zipper::concepts::IndexPackTuple T, rank_type... ranks>
  auto _coeff_ref(const T &idxs, std::integer_sequence<rank_type, ranks...>)
      -> value_type &
    requires(traits::is_assignable())
  {
    return expression().coeff_ref(std::get<ranks>(idxs)...);
  }

  template <zipper::concepts::IndexPackTuple T, rank_type... ranks>
  auto _const_coeff_ref(const T &idxs,
                        std::integer_sequence<rank_type, ranks...>) const
      -> const value_type &
    requires(traits::is_referrable())
  {
    return expression().const_coeff_ref(std::get<ranks>(idxs)...);
  }
};

} // namespace unary
} // namespace zipper::expression
#endif
