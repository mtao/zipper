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
#include <fmt/std.h>
#pragma GCC diagnostic pop
#include <vector>

#include "extents/extents_formatter.hpp"
#include "zipper/Vector.hpp"
#include "zipper/concepts/MatrixBaseDerived.hpp"
#include "zipper/concepts/VectorBaseDerived.hpp"
#include "zipper/types.hpp"

namespace zipper {

template <concepts::MatrixBaseDerived Mat>
auto format_as(Mat const& M) {
    constexpr static index_type dim =
        std::decay_t<decltype(M)>::static_extent(0);
    if constexpr (dim == std::dynamic_extent) {
        auto f = [M]<std::size_t... N>(std::index_sequence<N...>) {
            return std::make_tuple(M.row(N)...);
        };
        return fmt::format("{}",
                           fmt::join(f(std::make_index_sequence<dim>{}), ","));
    } else {
        using T = Mat::value_type;
        constexpr static index_type R = Mat::extents_type::static_extent(1);
        std::vector<Vector<T, R>> a;
        // std::vector<std::string> a_;
        for (index_type j = 0; j < M.extent(0); ++j) {
            a.emplace_back(M.row(j));
        }
        return fmt::format("[{}]", fmt::join(a, ";"));
    }
}
}  // namespace zipper
template <zipper::concepts::VectorBaseDerived Vec>
    requires(
        !std::is_same_v<std::decay_t<Vec>,
                        zipper::Vector<typename Vec::value_type,
                                       Vec::extents_type::static_extent(0)>>)
struct fmt::formatter<Vec>
    : formatter<zipper::Vector<typename Vec::value_type,
                               Vec::extents_type::static_extent(0)>> {
    // parse is inherited from formatter<string_view>.

    auto format(const Vec& v, format_context& ctx) const
        -> format_context::iterator {
        return formatter<zipper::Vector<typename Vec::value_type,
                                        Vec::extents_type::static_extent(0)>>::
            format(v.eval(), ctx);
    }
};
#endif
