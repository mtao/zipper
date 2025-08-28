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
#include "as_array.hpp"
#include "zipper/types.hpp"

namespace zipper::utils::extents {}

#if defined(__cpp_lib_mdspan)
namespace std {
#else
namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
#endif
template <typename index_type, index_type... Extents>
std::string format_as(const extents<index_type, Extents...>& foo) {
    return fmt::format("extents({})",
                       fmt::join(zipper::utils::extents::as_array(foo), ","));
}
// this last macro is just to prevent clang-format from changing the comment
// when committing on different machines
#if defined(__cpp_lib_mdspan)
}  // namespace std
#else
}  // namespace MDSPAN_IMPL_STANDARD_NAMESPACE
#endif

#if defined(ASDF)

#include <format>
template <typename index_type, index_type... Extents>
struct std::formatter<
    MDSPAN_IMPL_STANDARD_NAMESPACE ::extents<index_type, Extents...>, char> {
    template <class ParseContext>
    constexpr ParseContext::iterator parse(ParseContext& ctx) {
        auto it = ctx.begin();
        // if (it == ctx.end())
        //     return it;

        // if (*it == '#')
        //{
        //     quoted = true;
        //     ++it;
        // }
        // if (it != ctx.end() && *it != '}')
        //     throw std::format_error("Invalid format args for
        //     QuotableString.");

        return it;
    }

    template <class FmtContext>
    FmtContext::iterator format(const MDSPAN_IMPL_STANDARD_NAMESPACE ::extents<
                                    index_type, Extents...>& s,
                                FmtContext& ctx) const {
        auto f = [s]<std::size_t... N>(std::index_sequence<N...>) {
            return std::make_tuple(s.extent(N)...);
        };
        return std::format_to(
            ctx.out(), "extents({})",
            f(std::make_index_sequence<sizeof...(Extents)>{}));
    }
};
#endif

#endif
