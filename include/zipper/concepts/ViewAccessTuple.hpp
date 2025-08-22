

#if !defined(ZIPPER_CONCEPTS_VIEWACCESSTUPLE_HPP)
#define ZIPPER_CONCEPTS_VIEWACCESSTUPLE_HPP
#include <type_traits>

#include "IndexPackLike.hpp"
#include "SlicePackLike.hpp"
#include "TupleLike.hpp"
#include "ViewAccessPack.hpp"

namespace zipper::concepts {
namespace detail {

template <typename... T>
struct view_access_tuple : public std::false_type {};

template <TupleLike T>
struct view_access_tuple<T> {
    template <std::size_t... N>
    constexpr static bool value_temp(std::index_sequence<N...>) {
        return ViewAccessPack<std::tuple_element_t<N, T>...>;
    }

    constexpr static bool value =
        value_temp(std::make_index_sequence<std::tuple_size_v<T>>{});
};

}  // namespace detail
// defines whether a tuple can be 
template <typename T>
concept ViewAccessTuple = detail::view_access_tuple<std::decay_t<T>>::value;

}  // namespace zipper::concepts
#endif
