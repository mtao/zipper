#if !defined(ZIPPER_VIEWS_DETAIL_ASSIGNABLE_EXTENTS_HPP)
#define ZIPPER_VIEWS_DETAIL_ASSIGNABLE_EXTENTS_HPP
#include "zipper/concepts/ExtentsType.hpp"
#include "zipper/types.hpp"
namespace zipper::detail::extents {

// assigns the first parameter into the second parameter
template <typename, typename>
struct assignable_extents;
template <index_type... FromIndices, index_type... ToIndices>
    requires(sizeof...(ToIndices) == sizeof...(FromIndices))
struct assignable_extents<zipper::extents<FromIndices...>,
                          zipper::extents<ToIndices...>> {
    constexpr static bool value =
        ((FromIndices != std::dynamic_extent && ToIndices != std::dynamic_extent
              ? FromIndices == ToIndices
              : true) &&
         ...);

    template <rank_type... I>
    constexpr static bool value_runtime(
        const zipper::extents<FromIndices...>& o,
        std::integer_sequence<rank_type, I...>) {
        return ((FromIndices == std::dynamic_extent &&
                         ToIndices != std::dynamic_extent
                     ? ToIndices == o.extent(I)
                     : true) &&
                ... && value);
    }
    constexpr static bool value_runtime(const zipper::extents<FromIndices...>& o) {
        return value_runtime(
            o, std::make_integer_sequence<rank_type, sizeof...(FromIndices)>{});
    }
};

template <index_type... ToIndices>
struct assignable_extents<zipper::extents<>, zipper::extents<ToIndices...>>
    : public std::true_type {};
template <index_type... FromIndices, index_type... ToIndices>
    requires(sizeof...(ToIndices) != sizeof...(FromIndices))
struct assignable_extents<zipper::extents<FromIndices...>,
                          zipper::extents<ToIndices...>> : public std::false_type {
};

template <concepts::ExtentsType From, concepts::ExtentsType To>
constexpr static bool assignable_extents_v =
    assignable_extents<From, To>::value;

}  // namespace zipper::detail::extents
#endif
