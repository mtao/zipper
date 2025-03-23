#if !defined(UVL_DETAIL_EXTENTS_ALL_EXTENT_INDICES_HPP)
#define UVL_DETAIL_EXTENTS_ALL_EXTENT_INDICES_HPP

#include <range/v3/algorithm/copy.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/cartesian_product.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/transform.hpp>

#include "uvl/types.hpp"

namespace uvl::detail::extents{

template <index_type... SIndices, index_type... Indices>
auto all_extents_indices(const uvl::extents<SIndices...>& extents,
                         std::integer_sequence<index_type, Indices...>) {
    return ranges::views::cartesian_product(
        ranges::views::iota(index_type(0), extents.extent(Indices))...);
}

template <index_type... indices>
auto all_extents_indices(const uvl::extents<indices...>& extents) {
    return all_extents_indices(
        extents, std::make_integer_sequence<index_type, sizeof...(indices)>{});
}

template <index_type... indices>
auto all_extents_indices() {
    return all_extents_indices(uvl::extents<indices...>{});
}
template <typename... Args>
auto all_extents_indices(Args... indices) {
    return all_extents_indices(dextents<sizeof...(Args)>{indices...});
}
}  // namespace uvl::detail
#endif
