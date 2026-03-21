#if !defined(ZIPPER_EXPRESSION_UNARY_DIAGONALEXTRACT_HPP)
#define ZIPPER_EXPRESSION_UNARY_DIAGONALEXTRACT_HPP

#include "UnaryExpressionBase.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/assert.hpp"
#include "zipper/expression/detail/AssignHelper.hpp"

namespace zipper::expression {
namespace unary {
// extracts the diagonal elements of a tensor
template <zipper::concepts::QualifiedExpression ExpressionType> class DiagonalExtract;

} // namespace unary

/// Implementation details for DiagonalExtract expressions.
///
/// Holds child traits alias, child extents types, and the min-extent
/// computation functions. The class body uses child_traits to access the
/// child's extents_type and get_min_extent() to determine the diagonal
/// length at runtime.
template <zipper::concepts::QualifiedExpression ExpressionType>
struct detail::ExpressionDetail<unary::DiagonalExtract<ExpressionType>> {
  using child_traits = detail::ExpressionTraits<std::decay_t<ExpressionType>>;
  using value_type = typename child_traits::value_type;
  using base_extents_type = typename child_traits::extents_type;
  using base_extents_traits = zipper::detail::ExtentsTraits<base_extents_type>;

  template <std::size_t... Indices>
  constexpr static auto
  get_min_extent_static(std::integer_sequence<index_type, Indices...>)
      -> index_type {
    if constexpr (sizeof...(Indices) == 1) {
      return 1;
    } else {
      return std::min({base_extents_type::static_extent(Indices)...});
    }
  }

  constexpr static auto get_min_extent_static() -> index_type {
    if constexpr (base_extents_traits::is_dynamic) {
      return std::dynamic_extent;
    } else {
      return get_min_extent_static(
          std::make_integer_sequence<index_type, base_extents_type::rank()>{});
    }
  }

  static auto get_min_extent(const base_extents_type &e) -> index_type {
    if constexpr (base_extents_traits::is_static) {
      return get_min_extent_static();
    } else if constexpr (base_extents_type::rank() == 1) {
      return std::min<index_type>(0, e.extent(0));
    } else {
      index_type min = std::numeric_limits<index_type>::max();
      ;
      for (rank_type j = 0; j < e.rank(); ++j) {
        min = std::min(min, e.extent(j));
      }
      return min;
    }
  }
};

template <zipper::concepts::QualifiedExpression ExpressionType>
struct detail::ExpressionTraits<unary::DiagonalExtract<ExpressionType>>
    : public zipper::expression::unary::detail::DefaultUnaryExpressionTraits<
          ExpressionType> {
  using _Detail = detail::ExpressionDetail<unary::DiagonalExtract<ExpressionType>>;
  using value_type = typename _Detail::value_type;
  constexpr static bool is_coefficient_consistent = false;
  constexpr static bool is_value_based = false;
  using extents_type = zipper::extents<_Detail::get_min_extent_static()>;
};

namespace unary {
template <zipper::concepts::QualifiedExpression ExpressionType>
class DiagonalExtract
    : public UnaryExpressionBase<DiagonalExtract<ExpressionType>, ExpressionType> {
public:
  using self_type = DiagonalExtract<ExpressionType>;
  using traits = zipper::expression::detail::ExpressionTraits<self_type>;
  using detail_type = zipper::expression::detail::ExpressionDetail<self_type>;
  using extents_type = traits::extents_type;
  using value_type = traits::value_type;
  using Base = UnaryExpressionBase<self_type, ExpressionType>;
  using Base::expression;
  using child_traits = typename detail_type::child_traits;
  using child_extents_type = typename child_traits::extents_type;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

  DiagonalExtract(const DiagonalExtract &o) : DiagonalExtract(o.expression()) {}
  DiagonalExtract(DiagonalExtract &&o) : DiagonalExtract(o.expression()) {}

  auto operator=(const DiagonalExtract &) -> DiagonalExtract & = delete;
  auto operator=(DiagonalExtract &&) -> DiagonalExtract & = delete;

  template <typename U>
    requires std::constructible_from<typename Base::storage_type, U &&>
  DiagonalExtract(U &&b) : Base(std::forward<U>(b)) {}

  /// Recursively deep-copy child so the result owns all data.
  auto make_owned() const {
      auto owned_child = expression().make_owned();
      return DiagonalExtract<const decltype(owned_child)>(
          std::move(owned_child));
  }

  constexpr auto extent([[maybe_unused]] rank_type i) const -> index_type {
    ZIPPER_ASSERT(i == 0);
    return detail_type::get_min_extent(expression().extents());
  }

  constexpr auto extents() const -> extents_type {
    return extents_traits::make_extents_from(*this);
  }

  template <rank_type K>
  auto get_index(zipper::concepts::IndexPackTuple auto const &a) const
      -> index_type {
    return std::get<0>(a);
  }

  template <zipper::concepts::IndexPackTuple T, rank_type... ranks>
  auto _coeff(const T &idxs, std::integer_sequence<rank_type, ranks...>) const
      -> value_type {
    return expression().coeff(get_index<ranks>(idxs)...);
  }
  template <zipper::concepts::IndexPackTuple T, rank_type... ranks>
  auto _coeff_ref(const T &idxs, std::integer_sequence<rank_type, ranks...>)
      -> value_type &
    requires(traits::is_assignable())
  {
    return expression().coeff_ref(get_index<ranks>(idxs)...);
  }

  template <zipper::concepts::IndexPackTuple T, rank_type... ranks>
  auto _const_coeff_ref(const T &idxs,
                        std::integer_sequence<rank_type, ranks...>) const
      -> const value_type &
    requires(traits::is_referrable())
  {
    return expression().const_coeff_ref(get_index<ranks>(idxs)...);
  }

  template <typename... Args> auto coeff(Args &&...idxs) const -> value_type {
    return _coeff(
        std::make_tuple(std::forward<Args>(idxs)...),
        std::make_integer_sequence<rank_type, child_extents_type::rank()>{});
  }
  template <typename... Args>
  auto coeff_ref(Args &&...idxs) -> value_type &
    requires(traits::is_assignable())
  {
    return _coeff_ref(
        std::make_tuple(std::forward<Args>(idxs)...),
        std::make_integer_sequence<rank_type, child_extents_type::rank()>{});
  }
  template <typename... Args>
  auto const_coeff_ref(Args &&...idxs) const -> const value_type &
    requires(traits::is_referrable())
  {
    return _const_coeff_ref(
        std::make_tuple(std::forward<Args>(idxs)...),
        std::make_integer_sequence<rank_type, child_extents_type::rank()>{});
  }

  template <zipper::concepts::Expression V>
  void assign(const V &v)
    requires(traits::is_assignable() &&
             extents_traits::template is_convertable_from<
                 typename zipper::expression::detail::ExpressionTraits<
                     V>::extents_type>())
  {
    expression::detail::AssignHelper<V, self_type>::assign(v, *this);
  }

};

template <zipper::concepts::QualifiedExpression ExpressionType>
DiagonalExtract(const ExpressionType &v) -> DiagonalExtract<const ExpressionType&>;

} // namespace unary
} // namespace zipper::expression
#endif
