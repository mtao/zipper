
#if !defined(UVL_DETAIL_DYNAMIC_EXTENT_INDICES_HPP)
#define UVL_DETAIL_DYNAMIC_EXTENT_INDICES_HPP

#include "uvl/types.hpp"

namespace uvl::detail {

// template <index_type... Indices>
// constexpr auto dynamic_extent_indices(index_type... Indices) {
//     using E = extents<Indices...>;
// }
template <typename T>
struct DynamicExtentIndices;

template <index_type... indices>
struct DynamicExtentIndices<extents<indices...>> {
    constexpr static auto _eval() {
        using E = extents<indices...>;
        std::array<rank_type, E::rank_dynamic()> r;
        rank_type idx = 0;
        for (rank_type j = 0; j < E::rank(); ++j) {
            if (E::static_extent(j) == std::dynamic_extent) {
                r[idx++] = j;
            }
        }
        return r;
    }

    constexpr static auto value = _eval();

    //
};

template <typename T>
constexpr static auto dynamic_extent_indices_t = DynamicExtentIndices<T>::value;

}  // namespace uvl::detail
#endif
