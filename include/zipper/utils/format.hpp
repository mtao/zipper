#if !defined(ZIPPER_UTILS_FORMATTER_HPP)
#define ZIPPER_UTILS_FORMATTER_HPP
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
#include <vector>

#include "extents/extents_formatter.hpp"
#include "zipper/concepts/MatrixBaseDerived.hpp"
#include "zipper/concepts/VectorBaseDerived.hpp"
#include "zipper/types.hpp"

namespace zipper {
auto format_as(concepts::VectorBaseDerived auto const& v) {
    return fmt::format("{}", fmt::join(v.eval(), ","));
}
auto format_as(concepts::MatrixBaseDerived auto const& M) {
    constexpr static index_type dim =
        std::decay_t<decltype(M)>::static_extent(0);
    if constexpr (dim == std::dynamic_extent) {
        auto f = [M]<std::size_t... N>(std::index_sequence<N...>) {
            return std::make_tuple(M.row(N)...);
        };
        return fmt::format("{}",
                           fmt::join(f(std::make_index_sequence<dim>{}), ","));
    } else {
        std::vector<decltype(M.row(0))> a;
        for (index_type j = 0; j < M.extent(0); ++j) {
            a.emplace_back(M.row(j));
        }
        return fmt::format("{}", fmt::join(a, ";"));
    }
}
}  // namespace zipper

#endif
