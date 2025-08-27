#if !defined(ZIPPER_UTILS_EXTENTS_EXTENTS_FORMATTER_HPP)
#define ZIPPER_UTILS_EXTENTS_EXTENTS_FORMATTER_HPP
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-compare"
#pragma GCC diagnostic ignored "-Winline"
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wswitch-enum"
#pragma GCC diagnostic ignored "-Wswitch-default"
#pragma GCC diagnostic ignored "-Wstrict-overflow"
#pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
#pragma GCC diagnostic ignored "-Weffc++"
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wunsafe-buffer-usage"
#else
#pragma GCC diagnostic ignored "-Wmultiple-inheritance"
#pragma GCC diagnostic ignored "-Wabi-tag"
#endif
#include <fmt/format.h>
#include <fmt/ranges.h>
#pragma GCC diagnostic pop

#include "zipper/types.hpp"

#if defined(__cpp_lib_mdspan)
namespace std{
#else
namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
#endif
template <typename index_type, index_type... Extents>
std::string format_as(const extents<index_type, Extents...>& foo) {
    auto f = [foo]<std::size_t... N>(std::index_sequence<N...>) {
        return std::make_tuple(foo.extent(N)...);
    };
    return fmt::format(
        "extents({})",
        fmt::join(f(std::make_index_sequence<sizeof...(Extents)>{}), ","));
}
// this last macro is just to prevent clang-format from changing the comment when committing on different machines
#if defined(__cpp_lib_mdspan)
}  // namespace std
#else
}  // namespace MDSPAN_IMPL_STANDARD_NAMESPACE
#endif

#endif
