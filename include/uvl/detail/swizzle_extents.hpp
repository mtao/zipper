#if !defined(UVL_DETAIL_SWIZZLE_EXTENTS_HPP)
#define UVL_DETAIL_SWIZZLE_EXTENTS_HPP

#include "uvl/types.hpp"

namespace uvl::detail {

template <typename, typename>
struct ExtentsSwizzler {};
template <index_type... Indices, index_type... SwizzleIndices>
struct ExtentsSwizzler<extents<Indices...>, extents<SwizzleIndices...>> {
    using destination_extents_type = extents<Indices...>;
    using swizzle_extents_type = extents<SwizzleIndices...>;
    using extents_type = extents<destination_extents_type::static_extent(SwizzleIndices)...>;
    static extents_type get_extent(const destination_extents_type& s) {
        //return 
        //    extents_type(s.extent(SwizzleIndices)...);
    }
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
template <index_type... SwizzledIndices, index_type... Indices>
auto swizzle_extents(const extents<Indices...>& a) {
    return ExtentsSwizzler<extents<Indices...>, extents<SwizzledIndices...>>(a);
}

//template <index_type... Indices, index_type... SIndices>
//auto merge_extents(const extents<Indices...>& a,
//const extents<SIndices...>& b
//        ) {
//    return ExtentsSwizzler<extents<Indices...>, A>::run(a);
//}
}  // namespace uvl::detail
#endif
