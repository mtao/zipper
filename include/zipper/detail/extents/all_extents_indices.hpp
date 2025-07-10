#if !defined(ZIPPER_DETAIL_EXTENTS_ALL_EXTENT_INDICES_HPP)
#define ZIPPER_DETAIL_EXTENTS_ALL_EXTENT_INDICES_HPP

#include <range/v3/algorithm/copy.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/cartesian_product.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/transform.hpp>

#include "zipper/types.hpp"

namespace zipper::detail::extents {

template <index_type... SIndices, index_type... Indices>
auto all_extents_indices(const zipper::extents<SIndices...>& extents,
                         std::integer_sequence<index_type, Indices...>) {
    return ranges::views::cartesian_product(
        ranges::views::iota(index_type(0), extents.extent(Indices))...);
}

// returns every index available in an extent
template <index_type... indices>
auto all_extents_indices(const zipper::extents<indices...>& extents) {
    return all_extents_indices(
        extents, std::make_integer_sequence<index_type, sizeof...(indices)>{});
}

// returns every index available in an extent (static only)
template <index_type... indices>
auto all_extents_indices() {
    return all_extents_indices(zipper::extents<indices...>{});
}

// get all extents from a set of extent sizes (TODO: set dimensions)
template <typename... Args>
auto all_extents_indices(Args... indices) {
    return all_extents_indices(dextents<sizeof...(Args)>{indices...});
}
}  // namespace zipper::detail::extents
#endif
