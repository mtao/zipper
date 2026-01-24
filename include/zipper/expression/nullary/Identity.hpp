#if !defined(ZIPPER_EXPRESSION_NULLARY_IDENTITY_HPP)
#define ZIPPER_EXPRESSION_NULLARY_IDENTITY_HPP

#include <utility>

#include "NullaryExpressionBase.hpp"
#include "zipper/detail/make_integer_range_sequence.hpp"
#include "zipper/detail/merge_integer_sequence.hpp"
#include "zipper/detail/pack_index.hpp"

namespace zipper::expression {
namespace nullary {

namespace detail {

template <std::size_t... N>
constexpr static auto
_indicesAllSame(zipper::concepts::IndexPackTuple auto const &t,
                std::index_sequence<N...>) -> bool {
  return (
      (std::get<N>(t) == std::get<zipper::detail::pack_index<0>(N...)>(t)) &&
      ...);
  //
}

constexpr static auto
indicesAllSame(zipper::concepts::IndexPackTuple auto const &t) -> bool {
  return _indicesAllSame(
      t,
      std::make_index_sequence<std::tuple_size_v<std::decay_t<decltype(t)>>>{});
}
} // namespace detail

template <typename T, index_type... Indices>
class Identity : public ExpressionBase<Identity<T, Indices...>>,
                 public zipper::extents<Indices...> {
public:
  using self_type = Identity<T, Indices...>;
  using nullary_base_type = NullaryExpressionBase<self_type>;

  using traits = zipper::expression::detail::ExpressionTraits<self_type>;
  using extents_type = traits::extents_type;
  using extents_traits = traits::extents_traits;
  using value_type = traits::value_type;

  using extents_type::extent;
  using extents_type::extents;
  using extents_type::rank;
  auto extents() const -> const extents_type & { return *this; }

  Identity(const Identity &) = default;
  Identity(Identity &&) = default;
  auto operator=(const Identity &) -> Identity & = default;
  auto operator=(Identity &&) -> Identity & = default;
  Identity()
    requires(extents_traits::is_static)
  = default;
  Identity(const extents_type &e) : extents_type(e) {}

  template <concepts::Index... Args>
  Identity(Args &&...args)
      : Identity(extents_type(std::forward<Args>(args)...)) {}

  /// Returns 1 if every input coefficient is the same
  auto coeff(concepts::Index auto &&...idxs) const -> value_type {
    if constexpr (sizeof...(idxs) == 1) {
      return ((idxs == 0) && ...);
    } else {
      return detail::indicesAllSame(std::make_tuple(idxs...)) ? 1 : 0;
    }
  }

private:
  // =====================================================
  // TODO: this should be used for sparse?
  template <rank_type R, concepts::Index... Args>
    requires(R < extents_traits::rank &&
             sizeof...(Args) == extents_traits::rank)
  constexpr auto nonZeros(Args &&...args) const -> std::vector<index_type> {
    if (_indicesAllSame(std::make_tuple(args...),
                        zipper::detail::combine_integer_sequence(
                            std::make_index_sequence<R>{},
                            zipper::detail::make_integer_range_sequence<
                                rank_type, R + 1, extents_traits::rank>()))) {
      if constexpr (R == 0 && extents_traits::rank > 1) {
        return {zipper::detail::pack_index<1>(args...)};
      } else {
        return {zipper::detail::pack_index<0>(args...)};
      }

    } else {
      return {};
    }
  }
};
} // namespace nullary

template <typename T, index_type... Indices>
struct detail::ExpressionTraits<nullary::Identity<T, Indices...>>
    : public BasicExpressionTraits<
          T, zipper::extents<Indices...>,
          expression::detail::AccessFeatures{
              .is_const = false, .is_reference = false, .is_alias_free = true},
          expression::detail::ShapeFeatures{.is_resizable = true}> {};
} // namespace zipper::expression

#endif
