

#if !defined(UVL_CONCEPTS_VIEWACCESSTUPLE_HPP)
#define UVL_CONCEPTS_VIEWACCESSTUPLE_HPP
#include <type_traits>

#include "IndexPackLike.hpp"
#include "SlicePackLike.hpp"
#include "TupleLike.hpp"

namespace uvl::concepts {
namespace detail {

template <typename... T>
struct view_access_tuple : public std::false_type {};

template <TupleLike T>
struct view_access_tuple<T> {
    template <std::size_t... N>
    constexpr static bool value_temp(std::index_sequence<N...>) {
        return (SliceLike<std::tuple_element_t<N, T>> && ...);
    }

    constexpr static bool value =
        value_temp(std::make_index_sequence<std::tuple_size_v<T>>{});
};

}  // namespace detail
template <typename T>
concept ViewAccessTuple = detail::view_access_tuple<T>::value;

}  // namespace uvl::concepts
#endif
