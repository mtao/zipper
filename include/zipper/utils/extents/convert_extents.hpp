#if !defined(ZIPPER_UTILS_EXTENTS_CONVERT_EXTENTS_HPP)
#define ZIPPER_UTILS_EXTENTS_CONVERT_EXTENTS_HPP

#include "assignable_extents.hpp"
#include "zipper/types.hpp"

namespace zipper::utils::extents {

template <typename, typename>
struct ConvertExtents {};
template <index_type... FromIndices, index_type... ToIndices>
struct ConvertExtents<zipper::extents<FromIndices...>, zipper::extents<ToIndices...>> {
    static zipper::extents<ToIndices...> run(const zipper::extents<FromIndices...>& s)
        requires(sizeof...(FromIndices) == sizeof...(ToIndices) &&
                 assignable_extents_v<zipper::extents<FromIndices...>,
                                      zipper::extents<ToIndices...>>)
    {
        constexpr static rank_type my_dynamic_rank =
            zipper::extents<ToIndices...>::rank_dynamic();

        if (!assignable_extents<zipper::extents<FromIndices...>,
                                zipper::extents<ToIndices...>>::value_runtime(s)) {
            throw std::runtime_error("Was unable to convert extent {} to {}");
        }
        if constexpr (my_dynamic_rank == 0) {
            return zipper::extents<ToIndices...>{};
        } else {
            std::array<index_type, my_dynamic_rank> arr;
            rank_type arr_idx = 0;
            for (rank_type j = 0; j < zipper::extents<FromIndices...>::rank(); ++j) {
                if (zipper::extents<ToIndices...>::static_extent(j) ==
                    std::dynamic_extent) {
                    arr[arr_idx++] = s.extent(j);
                }
            }
            return zipper::extents<ToIndices...>(arr);
        }
    }
};
template <concepts::ExtentsType A, index_type... ToIndices>
A convert_extents(const zipper::extents<ToIndices...>& a) {
    return ConvertExtents<zipper::extents<ToIndices...>, A>::run(a);
}

// template <index_type... ToIndices, index_type... FromIndices>
// auto merge_extents(const extents<ToIndices...>& a,
// const extents<FromIndices...>& b
//         ) {
//     return ConvertExtents<extents<ToIndices...>, A>::run(a);
// }
}  // namespace zipper::utils::extents
#endif
