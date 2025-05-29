#if !defined(ZIPPER_DETAIL_MERGE_INTEGER_SEQUENCE_HPP)
#define ZIPPER_DETAIL_MERGE_INTEGER_SEQUENCE_HPP

#include <type_traits>
#include <utility>

namespace zipper::detail {

template <typename A, typename B, A... N, B... M>
auto combine_integer_sequence(std::integer_sequence<A, N...>,
                              std::integer_sequence<B, M...>) {
    return std::integer_sequence<std::common_type_t<A, B>, N..., M...>{};
}
}  // namespace zipper::detail
#endif
