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
#include "zipper/concepts/Expression.hpp"
#include "zipper/types.hpp"

namespace zipper {

template <zipper::concepts::Expression ZType> auto format_as_(ZType const &M) {
  constexpr static zipper::rank_type rank = ZType::extents_type::rank();

  if constexpr (rank == 0) {
    return fmt::format("{}", M());
  } else if constexpr (rank == 1) {
    return fmt::format("{}", as_vector(M).eval());
  } else if constexpr (rank > 1) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-compare"
    auto partial_trace_ = []<zipper::rank_type... N>(
                              const ZType &V, zipper::index_type j,
                              std::integer_sequence<zipper::rank_type, N...>) {
      return V.slice(j, std::enable_if_t<(N == N), zipper::full_extent_t>{}...);
    };
#pragma GCC diagnostic pop
    auto partial_trace = [partial_trace_](const ZType &V,
                                          zipper::index_type j) {
      return partial_trace_(
          V, j, std::make_integer_sequence<zipper::rank_type, rank - 1>{});
    };
    std::vector<std::string> fmts;
    for (zipper::index_type j = 0; j < M.extent(0); ++j) {
      fmts.emplace_back(format_as_(partial_trace(M, j)));
    }
    return fmt::format("[{}]", fmt::join(fmts, ";"));
  }
}

} // namespace zipper

#endif
