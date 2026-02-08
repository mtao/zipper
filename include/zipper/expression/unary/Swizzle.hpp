#if !defined(ZIPPER_EXPRESSION_UNARY_SWIZZLEVIEW_HPP)
#define ZIPPER_EXPRESSION_UNARY_SWIZZLEVIEW_HPP

#include "UnaryExpressionBase.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/extents/swizzle_extents.hpp"
#include "zipper/expression/detail/AssignHelper.hpp"
#include "zipper/utils/extents/all_extents_indices.hpp"

namespace zipper::expression {
namespace unary {
template <zipper::concepts::QualifiedExpression ExpressionType,
          index_type... Indices>
class Swizzle;

} // namespace unary
template <zipper::concepts::QualifiedExpression QualifiedExprType,
          index_type... Indices>
struct detail::ExpressionTraits<
    unary::Swizzle<QualifiedExprType, Indices...>>
    : public zipper::expression::unary::detail::DefaultUnaryExpressionTraits<
          QualifiedExprType, true> {
  using ExprType = std::decay_t<QualifiedExprType>;
  using swizzler_type = zipper::detail::extents::ExtentsSwizzler<Indices...>;

  using Base = ExpressionTraits<QualifiedExprType>;
  using extents_type = swizzler_type::template extents_type_swizzler_t<
      typename Base::extents_type>;
  using value_type = Base::value_type;
  constexpr static bool is_coefficient_consistent = false;
  constexpr static bool is_value_based = false;
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
  using ExprType = traits::ExprType;
  using extents_type = traits::extents_type;
  using value_type = traits::value_type;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
  using swizzler_type = traits::swizzler_type;
  using Base = UnaryExpressionBase<self_type, QualifiedExprType>;
  using Base::extent;
  using Base::expression;
  constexpr static rank_type internal_rank = ExprType::extents_type::rank();
  constexpr static std::array<rank_type, internal_rank>
      to_internal_rank_indices = swizzler_type::valid_internal_indices;

  Swizzle(const Swizzle &) = default;
  Swizzle(Swizzle &&) = default;
  auto operator=(const Swizzle &) -> Swizzle & = delete;
  auto operator=(Swizzle &&) -> Swizzle & = delete;
  Swizzle(QualifiedExprType &b)
      : Base(b, swizzler_type::swizzle_extents(b.extents())) {}

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
    static_assert(extents_type::rank() == sizeof...(Args));
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
