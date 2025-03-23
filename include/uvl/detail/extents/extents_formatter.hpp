#if !defined(UVL_DETAIL_EXTENTS_EXTENTS_FORMATTER_HPP)
#define UVL_DETAIL_EXTENTS_EXTENTS_FORMATTER_HPP
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-compare"
#include <fmt/format.h>
#include <fmt/ranges.h>
#pragma GCC diagnostic pop

#include "uvl/types.hpp"

namespace std::experimental {
template <typename index_type, index_type... Extents>
std::string format_as(const extents<index_type, Extents...>& foo) {
    auto f = [foo]<std::size_t... N>(std::index_sequence<N...>) {
        return std::make_tuple(foo.extent(N)...);
    };
    return fmt::format(
        "extents({})",
        fmt::join(f(std::make_index_sequence<sizeof...(Extents)>{}), ","));
}
};  // namespace std::experimental

#endif
