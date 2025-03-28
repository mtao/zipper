#if !defined(ZIPPER_VIEWS_DETAIL_SEQUENCE_TO_ARRAY_HPP)
#define ZIPPER_VIEWS_DETAIL_SEQUENCE_TO_ARRAY_HPP
#include <utility>

#include "zipper/types.hpp"
namespace zipper::detail::extents {

// assigns the first parameter into the second parameter
template <typename>
struct sequence_to_array;
template <typename T, T... Indices>
struct sequence_to_array<std::integer_sequence<T, Indices...>> {
    constexpr static std::array<T, sizeof...(Indices)> value =
        std::array<T, sizeof...(Indices)>{{Indices...}};
};
template <index_type... Indices>
struct sequence_to_array<zipper::extents<Indices...>> {
    constexpr static std::array<index_type, sizeof...(Indices)> value =
        std::array<index_type, sizeof...(Indices)>{{Indices...}};
};

template <typename T>
constexpr static auto sequence_to_array_v = sequence_to_array<T>::value;

}  // namespace zipper::detail::extents
#endif
