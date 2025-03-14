
#if !defined(UVL_DETAIL_ALL_TUPLE_SIZE_HPP)
#define UVL_DETAIL_ALL_TUPLE_SIZE_HPP

#include <range/v3/utility/common_tuple.hpp>
#include <tuple>

namespace uvl::detail {

template <typename T>
struct tuple_size : public std::tuple_size<T> {};

template <typename... Types>
struct tuple_size<ranges::common_tuple<Types...>> {
    constexpr static std::size_t value = sizeof...(Types);
};

template <typename T>
using tuple_size_v = tuple_size<T>::value;

}  // namespace uvl::detail
#endif
