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

// template <zipper::concepts::MatrixBaseDerived Mat>
// auto format_as(Mat const& M) {
//     return 0;
// }
namespace zipper {

// auto format_as(auto const& M) { return fmt::format("{}", M.view()); }
// template <typename T, zipper::index_type N>
// auto format_as(
//    const zipper::storage::PlainObjectStorage<T, zipper::extents<N>>& v) {
//    return fmt::format("{}", v.as_std_span());
//}
// template <typename T, zipper::index_type N>
// auto format_as(const zipper::storage::SpanStorage<T, zipper::extents<N>>& v)
// {
//    return fmt::format("{}", v.as_std_span());
//}

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
// template <zipper::concepts::ZipperBaseDerived ZType>
// auto format_as(ZType const& M) {
//     return format_as_(static_cast<typename ZType::Base&>(M));
// }
// template <zipper::concepts::Expression ZType>
//  requires(
//      !std::is_same_v<ZType, Vector<typename ZType::value_type,
//                                    ZType::extents_type::static_extent(0)>>)
// auto format_as(ZType const &M) {
//  return format_as_(M);
//}
// template <zipper::concepts::MatrixBaseDerived ZType>
// auto format_as(ZType const& M) {
//     return format_as_(M);
// }
//  template <typename T, index_type R>
//  auto format_as(Vector<T, R> const& M) {
//     return fmt::format("Hi {}", M.extents());
//     // return format_as_(M);
// }
//  template <typename T, index_type R, index_type C>
//  auto format_as(Matrix<T, R, C> const& M) {
//     return format_as_(M);
// }
} // namespace zipper

#endif
