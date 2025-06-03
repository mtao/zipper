
#if !defined(ZIPPER_UTILS_DETAIL_TUPLE_TO_ARRAY_HPP)
#define ZIPPER_UTILS_DETAIL_TUPLE_TO_ARRAY_HPP

#include "zipper/concepts/TupleLike.hpp"

namespace zipper::utils::detail {
template <zipper::concepts::TupleLike T, size_t... N>
auto _tuple_to_array(const T& t, std::index_sequence<N...>) {
    return std::array<std::common_type_t<std::tuple_element_t<N, T>...>,
                      std::tuple_size_v<T>>{{std::get<N>(t)...}};
}

template <zipper::concepts::TupleLike T>
auto tuple_to_array(const T& t) {
    return _tuple_to_array(t, std::make_index_sequence<std::tuple_size_v<T>>{});
}
}  // namespace zipper::utils::detail
#endif
