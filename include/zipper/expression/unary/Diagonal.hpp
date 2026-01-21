#if !defined(ZIPPER_expression_UNARY_DIAGONALVIEW_HPP)
#define ZIPPER_expression_UNARY_DIAGONALVIEW_HPP

#include "UnaryExpressionBase.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/expression/SizedExpressionBase.hpp"
#include "zipper/expression/detail/AssignHelper.hpp"

namespace zipper::expression {
namespace unary {
// extracts the diagonal elements of a tensor
template <zipper::concepts::QualifiedExpression ExpressionType> class Diagonal;

} // namespace unary
template <zipper::concepts::QualifiedExpression ExpressionType>
struct detail::ExpressionTraits<unary::Diagonal<ExpressionType>>
    : public zipper::expression::unary::detail::DefaultUnaryExpressionTraits<
          ExpressionType> {
  using Base = detail::ExpressionTraits<ExpressionType>;
  using value_type = Base::value_type;
  using base_extents_type = Base::extents_type;
  using base_extents_traits = zipper::detail::ExtentsTraits<base_extents_type>;
  constexpr static bool is_coefficient_consistent = false;
  constexpr static bool is_assignable = true;

  //
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
  using extents_type = zipper::extents<get_min_extent_static()>;
  static auto get_extents(const base_extents_type &e) -> extents_type {
    if constexpr (base_extents_traits::is_static) {
      return {};
    } else {
      return extents_type(get_min_extent(e));
    }
  }
};

namespace unary {
template <zipper::concepts::QualifiedExpression ExpressionType>
class Diagonal
    : public UnaryExpressionBase<Diagonal<ExpressionType>, ExpressionType> {
public:
  using self_type = Diagonal<ExpressionType>;
  using traits = zipper::expression::detail::ExpressionTraits<self_type>;
  using extents_type = traits::extents_type;
  using value_type = traits::value_type;
  using Base = UnaryExpressionBase<self_type, ExpressionType>;
  using Base::extent;
  using Base::view;
  using view_traits = traits::Base;
  using view_extents_type = view_traits::extents_type;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

  constexpr static bool holds_extents = traits::holds_extents;
  static_assert(holds_extents);

  constexpr static std::array<rank_type, view_extents_type::rank()>
      actionable_indices = traits::actionable_indices;

  Diagonal(const Diagonal &o) : Diagonal(o.view()) {}
  Diagonal(Diagonal &&o) : Diagonal(o.view()) {}

  auto operator=(const Diagonal &) -> Diagonal & = delete;
  auto operator=(Diagonal &&) -> Diagonal & = delete;

  Diagonal(ExpressionType &b) : Base(b, traits::get_extents(b.extents())) {}

  template <rank_type K>
  auto get_index(zipper::concepts::IndexPackTuple auto const &a) const
      -> index_type {
    return std::get<0>(a);
  }

  template <zipper::concepts::IndexPackTuple T, rank_type... ranks>
  auto _coeff(const T &idxs, std::integer_sequence<rank_type, ranks...>) const
      -> value_type {
    return view().coeff(get_index<ranks>(idxs)...);
  }
  template <zipper::concepts::IndexPackTuple T, rank_type... ranks>
  auto _coeff_ref(const T &idxs, std::integer_sequence<rank_type, ranks...>)
      -> value_type &
    requires(traits::is_writable)
  {
    return view().coeff_ref(get_index<ranks>(idxs)...);
  }

  template <zipper::concepts::IndexPackTuple T, rank_type... ranks>
  auto _const_coeff_ref(const T &idxs,
                        std::integer_sequence<rank_type, ranks...>) const
      -> const value_type &
    requires(traits::is_writable)
  {
    return view().const_coeff_ref(get_index<ranks>(idxs)...);
  }

  template <typename... Args> auto coeff(Args &&...idxs) const -> value_type {
    return _coeff(
        std::make_tuple(std::forward<Args>(idxs)...),
        std::make_integer_sequence<rank_type, view_extents_type::rank()>{});
  }
  template <typename... Args>
  auto coeff_ref(Args &&...idxs) -> value_type &
    requires(traits::is_writable)
  {
    return _coeff_ref(
        std::make_tuple(std::forward<Args>(idxs)...),
        std::make_integer_sequence<rank_type, view_extents_type::rank()>{});
  }
  template <typename... Args>
  auto const_coeff_ref(Args &&...idxs) const -> const value_type &
    requires(traits::is_writable)
  {
    return _const_coeff_ref(
        std::make_tuple(std::forward<Args>(idxs)...),
        std::make_integer_sequence<rank_type, view_extents_type::rank()>{});
  }

  template <zipper::concepts::Expression V>
  void assign(const V &v)
    requires(traits::is_writable &&
             extents_traits::template is_convertable_from<
                 typename zipper::expression::detail::ExpressionTraits<
                     V>::extents_type>())
  {
    expression::detail::AssignHelper<V, self_type>::assign(v, *this);
  }
};

template <zipper::concepts::QualifiedExpression ExpressionType>
Diagonal(const ExpressionType &v) -> Diagonal<ExpressionType>;

} // namespace unary
} // namespace zipper::expression
#endif
