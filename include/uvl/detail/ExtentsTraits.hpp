#if !defined(UVL_STORAGE_DETAIL_STORAGE_TRAITS_HPP)
#define UVL_STORAGE_DETAIL_STORAGE_TRAITS_HPP

#include "uvl/types.hpp"

namespace uvl::detail {
// template <uvl::index_type... Extents>
// struct StorageExtentsTraits {
//     using extents_type = uvl::template extents_type<Extents...>;
//     using is_dynamic = extents_type::rank_dynamic();
//     using is_static = extents_type::rank_static();
//     using rank = extents_type::rank;
// };

template <typename>
struct extent_values;
template <index_type... Idxs>
struct extent_values<extents<Idxs...>> {
    constexpr static index_type static_size =
        ((Idxs == std::dynamic_extent ? 1 : Idxs) * ... * 1);
};

template <typename Extents>
struct ExtentsTraits {
    using extents_type = Extents;
    constexpr static rank_type rank = extents_type::rank();
    constexpr static rank_type rank_dynamic = extents_type::rank_dynamic();
    constexpr static rank_type rank_static = rank - rank_dynamic;
    constexpr static bool is_dynamic = rank_dynamic != 0;
    constexpr static bool is_static = !is_dynamic;

    constexpr static index_type static_size =
        extent_values<extents_type>::static_size;
    constexpr static index_type _size =
        extent_values<extents_type>::static_size;

    template <typename T>
    using span_type = std::conditional_t<is_static, std::span<T, static_size>,
                                         std::span<T, std::dynamic_extent>>;

    static constexpr index_type size(const extents_type& e) {
        if constexpr (is_static) {
            return static_size;
        } else {
            index_type s = 1;
            for (rank_type j = 0; j < e.rank(); ++j) {
                s *= e.extent(j);
            }
            return s;
        }
    }

    consteval bool is_dynamic_extent(rank_type i) {
        return extents_type::static_extent(i) == std::dynamic_extent;
    }
};
}  // namespace uvl::detail
#endif
