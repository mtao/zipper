#if !defined(ZIPPER_UTILS_FORMATTER_HPP)
#define ZIPPER_UTILS_FORMATTER_HPP
#include "zipper/detail/fmt.hpp"
#include <vector>

#include "extents/extents_formatter.hpp"
#include "zipper/concepts/Zipper.hpp"
#include "zipper/types.hpp"

namespace zipper {

// Internal helper — formats any Zipper object as a string.
// Not named format_as to avoid fmt's ADL format_as protocol
// (which only supports arithmetic return types in fmt 11).
template <zipper::concepts::Zipper ZType>
auto zipper_format_string(ZType const &M) -> std::string {
  using expression_type = typename ZType::expression_type;
  constexpr static zipper::rank_type rank =
      expression_type::extents_type::rank();

  if constexpr (rank == 0) {
    return fmt::format("{}", M());
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
    return fmt::format("{}", vals);
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
      fmts.emplace_back(zipper_format_string(partial_trace(M, j)));
    }
    return fmt::format("[{}]", fmt::join(fmts, ";"));
  }
}

} // namespace zipper

// Disable fmt's range formatting for all Zipper types so that
// Vector and Container (which have begin/end) don't get picked up
// by fmt/ranges.h before our formatter specialization.
template <typename T, typename Char>
struct fmt::range_format_kind<
    T, Char,
    std::enable_if_t<zipper::concepts::Zipper<T>>>
    : std::integral_constant<fmt::range_format, fmt::range_format::disabled> {};

// Provide fmt::formatter for all Zipper types.
template <typename T, typename Char>
struct fmt::formatter<
    T, Char,
    std::enable_if_t<zipper::concepts::Zipper<T>>>
    : fmt::formatter<std::string, Char> {
  template <typename FormatContext>
  auto format(const T &value, FormatContext &ctx) const -> decltype(ctx.out()) {
    return fmt::formatter<std::string, Char>::format(
        zipper::zipper_format_string(value), ctx);
  }
};

#endif
