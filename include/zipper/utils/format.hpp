#if !defined(ZIPPER_UTILS_FORMATTER_HPP)
#define ZIPPER_UTILS_FORMATTER_HPP
#include <format>
#include <string>
#include <vector>

#include "extents/extents_formatter.hpp"
#include "zipper/concepts/Zipper.hpp"
#include "zipper/types.hpp"

namespace zipper {

// Internal helper — formats any Zipper object as a string.
template <zipper::concepts::Zipper ZType>
auto zipper_format_string(ZType const &M) -> std::string {
  using expression_type = typename ZType::expression_type;
  constexpr static zipper::rank_type rank =
      expression_type::extents_type::rank();

  if constexpr (rank == 0) {
    return std::format("{}", M());
  } else if constexpr (rank == 1) {
    // Collect into std::vector to avoid infinite recursion:
    // formatting a Vector would re-enter this function because
    // Vector is also a Zipper type.
    using value_type = typename ZType::value_type;
    std::vector<value_type> vals;
    vals.reserve(static_cast<std::size_t>(M.extent(0)));
    for (zipper::index_type i = 0; i < M.extent(0); ++i) {
      vals.push_back(M(i));
    }
    return std::format("{}", vals);
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
    std::string result = "[";
    for (zipper::index_type j = 0; j < M.extent(0); ++j) {
      if (j > 0) result += ';';
      result += zipper_format_string(partial_trace(M, j));
    }
    result += ']';
    return result;
  }
}

} // namespace zipper

// Disable std's range formatting for all Zipper types so that
// Vector and DataArray (which have begin/end) don't get picked up
// by the range formatter before our formatter specialization.
template <typename T>
  requires zipper::concepts::Zipper<T>
constexpr auto std::format_kind<T> = std::range_format::disabled;

// Provide std::formatter for all Zipper types.
template <zipper::concepts::Zipper T>
struct std::formatter<T> : std::formatter<std::string> {
  auto format(const T &value, std::format_context &ctx) const
      -> std::format_context::iterator {
    return std::formatter<std::string>::format(
        zipper::zipper_format_string(value), ctx);
  }
};

#endif
