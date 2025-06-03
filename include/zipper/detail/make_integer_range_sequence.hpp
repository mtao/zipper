#if !defined(ZIPPER_DETAIL_MAKE_INTEGER_RANGE_SEQUENCE_HPP)
#define ZIPPER_DETAIL_MAKE_INTEGER_RANGE_SEQUENCE_HPP

#include <type_traits>
#include <utility>

namespace zipper::detail {

template <typename A, A Start, A End>
auto make_integer_range_sequence() {
    auto r = []<A... R>(std::integer_sequence<A, R...>) {
        return std::integer_sequence<A, (R + Start)...>{};
    };
    return r(std::make_integer_sequence<A, End - Start>{});
}
}  // namespace zipper::detail
#endif
