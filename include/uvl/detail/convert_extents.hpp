
#if !defined(UVL_DETAIL_CONVERT_EXTENTS_HPP)
#define UVL_DETAIL_CONVERT_EXTENTS_HPP

#include "uvl/types.hpp"

namespace uvl::detail {

template <typename, typename>
struct ConvertExtents {};
template <index_type... SIndices, index_type... Indices>
struct ConvertExtents<extents<SIndices...>, extents<Indices...>> {
    static extents<Indices...> run(const extents<SIndices...>& s) {
        std::array<index_type, extents<Indices...>::rank_dynamic()> arr;
        rank_type arr_idx = 0;
        for (rank_type j = 0; j < extents<SIndices...>::rank(); ++j) {
            if (extents<Indices...>::static_extent(j) == std::dynamic_extent) {

                arr[arr_idx++] = s.extent(j);
            }
        }
        return extents<Indices...>(arr);
    }
};
template <typename A, index_type... Indices>
A convert_extents(const extents<Indices...>& a) {
    return ConvertExtents<extents<Indices...>, A>::run(a);
}
}  // namespace uvl::detail
#endif
