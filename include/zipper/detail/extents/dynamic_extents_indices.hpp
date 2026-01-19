#if !defined(ZIPPER_DETAIL_EXTENTS_DYNAMIC_EXTENTS_INDICES_HPP)
#define ZIPPER_DETAIL_EXTENTS_DYNAMIC_EXTENTS_INDICES_HPP
#include <iostream>

#include "zipper/types.hpp"

namespace zipper::detail::extents {

// template <index_type... Indices>
// constexpr auto dynamic_extent_indices(index_type... Indices) {
//     using E = extents<Indices...>;
// }
template <typename T> struct DynamicExtentIndices;

template <index_type... indices>
struct DynamicExtentIndices<zipper::extents<indices...>> {
  using E = zipper::extents<indices...>;
  // the number of dynamic indices held in the given extents.
  // i.e the number of std::dynamic_extent in indices...
  constexpr static rank_type rank_dynamic = E::rank_dynamic();

  // returns the number of indices
  consteval static auto _eval() {
    std::array<rank_type, rank_dynamic> r;
    rank_type idx = 0;
    for (rank_type j = 0; j < E::rank(); ++j) {
      if (E::static_extent(j) == std::dynamic_extent) {
        r[idx++] = j;
      }
    }
    return r;
  }
  // indices in the input extent that are dynamic
  // i.e E::static_extent(j) == 0 iff j is in dynamic_indices
  constexpr static auto dynamic_indices = _eval();
  constexpr static auto value = dynamic_indices;

  constexpr static rank_type dynamic_rank = sizeof...(indices);
  using dynamic_indices_type = std::array<rank_type, sizeof...(indices)>;

  // for an index J, if E::static_extent(J) == std::dynamic_extent returns the
  // index of that extent otherwise returns nothing
  template <rank_type J>
  consteval static auto get_dynamic_local_index() -> rank_type {
    for (rank_type j = 0; j < dynamic_indices.size(); ++j) {
      if (J == dynamic_indices[j]) {
        return j;
      }
    }
    assert(false);
    return rank_type{};
  }
  // composes the indices that belong to dynamic indices.
  // Note that due to swizzle the set of indices bieng
  template <std::size_t... Indices>
  consteval static auto
  get_dynamic_local_indices(std::index_sequence<Indices...>)
      -> const dynamic_indices_type {
    return std::array<rank_type, dynamic_rank>{
        {get_dynamic_local_index<Indices>()...}};
  }

  static consteval auto get_dynamic_local_indices() -> dynamic_indices_type {
    return get_dynamic_local_indices(
        std::make_integer_sequence<rank_type, E::rank()>{});
  }

  // maps from the full extents ranks to the local extents ranks
  constexpr static dynamic_indices_type dynamic_local_indices =
      get_dynamic_local_indices();

  template <std::size_t... N>
  static auto run(const zipper::extents<indices...> &e,
                  std::integer_sequence<std::size_t, N...>) {
    return std::array<index_type, rank_dynamic>{
        {e.extent(std::get<N>(value))...}};
  }
  static auto run(const zipper::extents<indices...> &e) {
    return run(e, std::make_index_sequence<std::size_t(rank_dynamic)>{});
  }

  //
};

template <typename T>
constexpr auto dynamic_extents_indices_v = DynamicExtentIndices<T>::value;
template <typename T, typename U = T> auto dynamic_extents(const T &extents) {
  return DynamicExtentIndices<U>::run(extents);
}

} // namespace zipper::detail::extents
#endif
