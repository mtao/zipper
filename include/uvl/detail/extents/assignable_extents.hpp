#if !defined(UVL_VIEWS_DETAIL_ASSIGNABLE_EXTENTS_HPP)
#define UVL_VIEWS_DETAIL_ASSIGNABLE_EXTENTS_HPP
#include "uvl/concepts/ExtentsType.hpp"
#include "uvl/types.hpp"
namespace uvl::detail::extents {

// assigns the first parameter into the second parameter
template <typename, typename>
struct assignable_extents;
template <index_type... FromIndices, index_type... ToIndices>
    requires(sizeof...(ToIndices) == sizeof...(FromIndices))
struct assignable_extents<uvl::extents<FromIndices...>,
                          uvl::extents<ToIndices...>> {
    constexpr static bool value =
        ((FromIndices != std::dynamic_extent && ToIndices != std::dynamic_extent
              ? FromIndices == ToIndices
              : true) &&
         ...);

    template <rank_type... I>
    constexpr static bool value_runtime(
        const uvl::extents<FromIndices...>& o,
        std::integer_sequence<rank_type, I...>) {
        return ((FromIndices == std::dynamic_extent &&
                         ToIndices != std::dynamic_extent
                     ? ToIndices == o.extent(I)
                     : true) &&
                ... && value);
    }
    constexpr static bool value_runtime(const uvl::extents<FromIndices...>& o) {
        return value_runtime(
            o, std::make_integer_sequence<rank_type, sizeof...(FromIndices)>{});
    }
};

template <index_type... ToIndices>
struct assignable_extents<uvl::extents<>, uvl::extents<ToIndices...>>
    : public std::true_type {};
template <index_type... FromIndices, index_type... ToIndices>
    requires(sizeof...(ToIndices) != sizeof...(FromIndices))
struct assignable_extents<uvl::extents<FromIndices...>,
                          uvl::extents<ToIndices...>> : public std::false_type {
};

template <concepts::ExtentsType From, concepts::ExtentsType To>
constexpr static bool assignable_extents_v =
    assignable_extents<From, To>::value;

}  // namespace uvl::detail::extents
#endif
