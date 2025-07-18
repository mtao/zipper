#if !defined(ZIPPER_UTILS_EXTENTS_ALL_EXTENT_INDICES_HPP)
#define ZIPPER_UTILS_EXTENTS_ALL_EXTENT_INDICES_HPP

// compressed_tuple is in use but deprecated, not sure how to stop it
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wpadded"

#pragma GCC diagnostic ignored "-Wunused-const-variable"
#pragma GCC diagnostic ignored "-Weffc++"
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wdeprecated-literal-operator"
#else
#pragma GCC diagnostic ignored "-Wnoexcept"
#pragma GCC diagnostic ignored "-Wuseless-cast"
#pragma GCC diagnostic ignored "-Wmultiple-inheritance"
#endif
#include <range/v3/algorithm/copy.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/cartesian_product.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/transform.hpp>
#pragma GCC diagnostic pop

#include "zipper/types.hpp"

namespace zipper::utils::extents {

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
}  // namespace zipper::utils::extents
#endif
