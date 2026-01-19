#if !defined(ZIPPER_UTILS_EXTENTS_INDICES_IN_RANGE_HPP)
#define ZIPPER_UTILS_EXTENTS_INDICES_IN_RANGE_HPP

#include "zipper/concepts/Extents.hpp"
#include "zipper/concepts/IndexArgument.hpp"
#include "zipper/detail/slice_helpers.hpp"
#include "zipper/types.hpp"

namespace zipper::utils::extents {

template <concepts::Extents Extents, concepts::IndexArgument... Indices,
          rank_type... N>
auto indices_in_range(std::integer_sequence<rank_type, N...>,
                      const Extents &extents, const Indices &...i) -> bool {
  auto check = []<rank_type J, typename T>(std::integral_constant<rank_type, J>,
                                           const Extents &e,
                                           const T &index) -> bool {
    if constexpr (concepts::Index<T>) {
      if (index_type(index) < 0) {
        return false;
      }
      if constexpr (Extents::static_extent(J) == std::dynamic_extent) {
        return index_type(index) < e.extent(J);
      } else {
        return index_type(index) < Extents::static_extent(J);
      }
    } else if constexpr (std::is_same_v<full_extent_t, std::decay_t<T>>) {
      return true;
    } else {
      constexpr index_type start = zipper::detail::first_of(index);
      constexpr index_type end = zipper::detail::last_of(index);

      if (0 < start) {
        return false;
      }

      if constexpr (Extents::static_extent(J) == std::dynamic_extent) {
        return index_type(end) < e.extent(J);
      } else {
        return index_type(end) < Extents::static_extent(J);
      }
    }
  };

  return (check(std::integral_constant<rank_type, N>{}, extents, i) && ...);
}

template <concepts::Extents Extents, concepts::Index... Indices>
auto indices_in_range(const Extents &e, const Indices &...i) -> bool {
  if constexpr (Extents::rank() == 0) {
    return true;
  } else {
    static_assert(Extents::rank() == sizeof...(Indices));
    return indices_in_range(
        std::make_integer_sequence<rank_type, Extents::rank()>{}, e, i...);
  }
}
} // namespace zipper::utils::extents
#endif
