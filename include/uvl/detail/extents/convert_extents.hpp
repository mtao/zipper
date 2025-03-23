#if !defined(UVL_DETAIL_EXTENTS_CONVERT_EXTENTS_HPP)
#define UVL_DETAIL_EXTENTS_CONVERT_EXTENTS_HPP

#include "assignable_extents.hpp"
#include "uvl/types.hpp"

namespace uvl::detail::extents {

template <typename, typename>
struct ConvertExtents {};
template <index_type... FromIndices, index_type... ToIndices>
struct ConvertExtents<uvl::extents<FromIndices...>, uvl::extents<ToIndices...>> {
    static uvl::extents<ToIndices...> run(const uvl::extents<FromIndices...>& s)
        requires(sizeof...(FromIndices) == sizeof...(ToIndices) &&
                 assignable_extents_v<uvl::extents<FromIndices...>,
                                      uvl::extents<ToIndices...>>)
    {
        constexpr static rank_type my_dynamic_rank =
            uvl::extents<ToIndices...>::rank_dynamic();

        if (!assignable_extents<uvl::extents<FromIndices...>,
                                uvl::extents<ToIndices...>>::value_runtime(s)) {
            throw std::runtime_error("Was unable to convert extent {} to {}");
        }
        if constexpr (my_dynamic_rank == 0) {
            return uvl::extents<ToIndices...>{};
        } else {
            std::array<index_type, my_dynamic_rank> arr;
            rank_type arr_idx = 0;
            for (rank_type j = 0; j < uvl::extents<FromIndices...>::rank(); ++j) {
                if (uvl::extents<ToIndices...>::static_extent(j) ==
                    std::dynamic_extent) {
                    arr[arr_idx++] = s.extent(j);
                }
            }
            return uvl::extents<ToIndices...>(arr);
        }
    }
};
template <concepts::ExtentsType A, index_type... ToIndices>
A convert_extents(const uvl::extents<ToIndices...>& a) {
    return ConvertExtents<uvl::extents<ToIndices...>, A>::run(a);
}

// template <index_type... ToIndices, index_type... FromIndices>
// auto merge_extents(const extents<ToIndices...>& a,
// const extents<FromIndices...>& b
//         ) {
//     return ConvertExtents<extents<ToIndices...>, A>::run(a);
// }
}  // namespace uvl::detail::extents
#endif
