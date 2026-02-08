#if !defined(ZIPPER_EXPRESSION_UNARY_HOMOGENEOUSVIEW_HPP)
#define ZIPPER_EXPRESSION_UNARY_HOMOGENEOUSVIEW_HPP

#include "UnaryExpressionBase.hpp"
#include "zipper/detail/pack_index.hpp"

namespace zipper::expression {
namespace detail {
template <rank_type N, index_type Size, typename>
struct extend_extents_single_dimension;
template <rank_type N, index_type Size, index_type... Idxs>
struct extend_extents_single_dimension<N, Size, extents<Idxs...>> {
  template <typename> struct _ {};
  template <rank_type... M>
  struct _<std::integer_sequence<rank_type, M...>> {
    template <rank_type S>
    constexpr static index_type V() {
      return zipper::detail::pack_index<S>(Idxs...);
    }
    using type =
        extents<N == M ? (V<M>() == std::dynamic_extent ? V<M>()
                                                        : V<M>() + Size)
                       : V<M>()...>;
  };
  using __ = _<std::make_integer_sequence<rank_type, sizeof...(Idxs)>>;
  using extended_extents_type = __::type;

  using dynamic_indices_helper =
      zipper::detail::extents::DynamicExtentIndices<extended_extents_type>;
  constexpr static rank_type rank_dynamic =
      dynamic_indices_helper::rank_dynamic;

  template <std::size_t... M>
  static auto run(const zipper::extents<Idxs...> &e,
                  std::integer_sequence<std::size_t, M...>) {
    constexpr static auto dynamic_indices = dynamic_indices_helper::value;
    return std::array<index_type, rank_dynamic>{
        {(e.extent(std::get<M>(dynamic_indices)) + (M == N ? Size : 0))...}};
  }
  static auto run(const zipper::extents<Idxs...> &e) {
    return run(e, std::make_index_sequence<std::size_t(rank_dynamic)>{});
  }
};
} // namespace detail

namespace unary {
enum class HomogeneousMode { Position, Vector, Affine };
template <HomogeneousMode Mode, zipper::concepts::Expression Child>
class Homogeneous;

} // namespace unary

template <unary::HomogeneousMode Mode, zipper::concepts::Expression Child>
struct detail::ExpressionTraits<unary::Homogeneous<Mode, Child>>
    : public zipper::expression::unary::detail::DefaultUnaryExpressionTraits<
          Child, true> {
  using ChildTraits = ExpressionTraits<Child>;
  using value_type = typename ChildTraits::value_type;

  using helper = detail::extend_extents_single_dimension<
      0, 1, typename ChildTraits::extents_type>;
  using extents_type = helper::extended_extents_type;

  static extents_type make_extents(const ChildTraits::extents_type &a) {
    return helper::run(a);
  }
  constexpr static bool holds_extents = true;
  // Homogeneous computes values (returns literal 0/1 for appended row) — not
  // referrable or assignable
  constexpr static zipper::detail::AccessFeatures access_features = {
      .is_const = true,
      .is_reference = false,
      .is_alias_free = ChildTraits::access_features.is_alias_free,
  };
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

namespace unary {
template <HomogeneousMode Mode, zipper::concepts::Expression Child>
class Homogeneous
    : public UnaryExpressionBase<Homogeneous<Mode, Child>, Child> {
public:
  using self_type = Homogeneous<Mode, Child>;
  using traits = zipper::expression::detail::ExpressionTraits<self_type>;
  using extents_type = traits::extents_type;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
  constexpr static bool is_static = extents_traits::is_static;
  using value_type = traits::value_type;

  using Base = UnaryExpressionBase<self_type, Child>;
  using Base::Base;
  using Base::expression;

  Homogeneous(Child &a)
    requires(is_static)
      : Base(a) {}
  Homogeneous(Child &a)
    requires(!is_static)
      : Base(a, traits::make_extents(a.extents())) {}

  template <typename... Args>
  value_type coeff(Args &&...args) const {
    if (index_type(zipper::detail::pack_index<0>(args...)) ==
        expression().extent(0)) {
      if constexpr (Mode == HomogeneousMode::Position) {
        return 1;
      } else if constexpr (Mode == HomogeneousMode::Vector) {
        return 0;
      } else {
        static_assert(extents_type::rank() == 2);
        if (index_type(zipper::detail::pack_index<1>(args...)) ==
            expression().extent(1) - 1) {
          return 1;
        } else {
          return 0;
        }
      }
    } else {
      return expression().coeff(args...);
    }
  }
};

template <zipper::concepts::Expression Child>
auto homogeneous(Child &a) {
  return Homogeneous<HomogeneousMode::Position, Child>(a);
}
template <zipper::concepts::Expression Child>
auto homogeneous_position(Child &a) {
  return Homogeneous<HomogeneousMode::Position, Child>(a);
}
template <zipper::concepts::Expression Child>
auto homogeneous_vector(Child &a) {
  return Homogeneous<HomogeneousMode::Vector, Child>(a);
}
template <zipper::concepts::Expression Child>
auto affine(Child &a) {
  return Homogeneous<HomogeneousMode::Affine, Child>(a);
}
} // namespace unary
} // namespace zipper::expression
#endif
