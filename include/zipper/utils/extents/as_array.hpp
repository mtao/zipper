#if !defined(ZIPPER_UTILS_EXTENTS_AS_ARRAY_HPP)
#define ZIPPER_UTILS_EXTENTS_AS_ARRAY_HPP

#include <array>

#include "zipper/types.hpp"

namespace zipper::utils::extents {

template <index_type... Extents>
std::array<index_type, sizeof...(Extents)> as_array(
    const zipper::extents<Extents...>& e) {
    auto f = []<std::size_t... N>(const auto& e_, std::index_sequence<N...>) {
        return std::array<index_type, sizeof...(Extents)>{{e_.extent(N)...}};
    };
    return f(e, std::make_index_sequence<sizeof...(Extents)>{});
}

}  // namespace zipper::utils::extents

#endif
