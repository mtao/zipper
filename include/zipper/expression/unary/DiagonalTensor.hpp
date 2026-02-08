#if !defined(ZIPPER_EXPRESSION_UNARY_DIAGONALTENSORVIEW_HPP)
#define ZIPPER_EXPRESSION_UNARY_DIAGONALTENSORVIEW_HPP

#include "UnaryExpressionBase.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/concepts/Index.hpp"
#include "zipper/expression/detail/AssignHelper.hpp"

namespace zipper::expression {
namespace unary {
// Creates an N-dimensional diagonal tensor from a 1D expression
template <zipper::concepts::QualifiedExpression ExpressionType, rank_type N>
  requires(ExpressionType::extents_type::rank() == 1)
class DiagonalTensor;

} // namespace unary
template <zipper::concepts::QualifiedExpression ExpressionType, rank_type N>
  requires(ExpressionType::extents_type::rank() == 1)
struct detail::ExpressionTraits<unary::DiagonalTensor<ExpressionType, N>>
    : public zipper::expression::unary::detail::DefaultUnaryExpressionTraits<
          ExpressionType, true> {
  using Base = detail::ExpressionTraits<ExpressionType>;
  using value_type = Base::value_type;
  using base_extents_type = Base::extents_type;
  using base_extents_traits = zipper::detail::ExtentsTraits<base_extents_type>;
  constexpr static bool is_coefficient_consistent = false;
  constexpr static bool is_value_based = false;
  constexpr static index_type base_size = base_extents_type::static_extent(0);

  template <typename> struct _detail;
  template <rank_type... Ns>
  struct _detail<std::integer_sequence<rank_type, Ns...>> {
    using extents_type = extents<(Ns * 0 + base_size)...>;

    constexpr static extents_type make_extents(const base_extents_type &e) {
      if constexpr (base_size == std::dynamic_extent) {
        return extents_type{(Ns * 0 + e.extent(0))...};
      } else {
        return {};
      }
    }
  };
  using _detail_type =
      _detail<decltype(std::make_integer_sequence<rank_type, N>{})>;

  using extents_type = _detail_type::extents_type;
  static constexpr extents_type make_extents(const base_extents_type &e) {
    return _detail_type::make_extents(e);
  }
};

namespace unary {
template <zipper::concepts::QualifiedExpression ExpressionType, rank_type N>
  requires(ExpressionType::extents_type::rank() == 1)
class DiagonalTensor
    : public UnaryExpressionBase<DiagonalTensor<ExpressionType, N>,
                                 ExpressionType> {
public:
  using self_type = DiagonalTensor<ExpressionType, N>;
  using traits = zipper::expression::detail::ExpressionTraits<self_type>;
  using extents_type = traits::extents_type;
  using value_type = traits::value_type;
  using Base = UnaryExpressionBase<self_type, ExpressionType>;
  using Base::extent;
  using Base::expression;
  using child_traits = traits::Base;
  using child_extents_type = child_traits::extents_type;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

  constexpr static bool holds_extents = traits::holds_extents;
  static_assert(holds_extents);

  DiagonalTensor(const DiagonalTensor &o)
      : DiagonalTensor(o.expression()) {}
  DiagonalTensor(DiagonalTensor &&o)
      : DiagonalTensor(o.expression()) {}

  auto operator=(const DiagonalTensor &)
      -> DiagonalTensor & = delete;
  auto operator=(DiagonalTensor &&)
      -> DiagonalTensor & = delete;

  DiagonalTensor(ExpressionType &b)
      : Base(b, traits::make_extents(b.extents())) {}

  template <rank_type K>
  index_type
  get_index(zipper::concepts::IndexPackTuple auto const &a) const {
    return std::get<0>(a);
  }

  template <zipper::concepts::IndexPackTuple T, rank_type... ranks>
  auto _coeff(const T &idxs,
              std::integer_sequence<rank_type, ranks...>) const -> value_type {
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

  template <typename... Args>
  value_type coeff(Args &&...idxs) const {
    return _coeff(
        std::make_tuple(std::forward<Args>(idxs)...),
        std::make_integer_sequence<rank_type,
                                   child_extents_type::rank()>{});
  }
  template <typename... Args>
  value_type &coeff_ref(Args &&...idxs)
    requires(traits::is_assignable())
  {
    return _coeff_ref(
        std::make_tuple(std::forward<Args>(idxs)...),
        std::make_integer_sequence<rank_type,
                                   child_extents_type::rank()>{});
  }
  template <typename... Args>
  const value_type &const_coeff_ref(Args &&...idxs) const
    requires(traits::is_referrable())
  {
    return _const_coeff_ref(
        std::make_tuple(std::forward<Args>(idxs)...),
        std::make_integer_sequence<rank_type,
                                   child_extents_type::rank()>{});
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

} // namespace unary
} // namespace zipper::expression
#endif
