#if !defined(ZIPPER_STORAGE_DETAIL_STORAGE_TRAITS_HPP)
#define ZIPPER_STORAGE_DETAIL_STORAGE_TRAITS_HPP

#include "zipper/utils/extents/convert_extents.hpp"
#include "extents/dynamic_extents_indices.hpp"
#include "zipper/types.hpp"

namespace zipper::detail {

template <typename>
struct extent_values;
template <index_type... Idxs>
struct extent_values<zipper::extents<Idxs...>> {
    constexpr static index_type static_sub_size =
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

    constexpr static index_type static_sub_size =
        extent_values<extents_type>::static_sub_size;

    constexpr static index_type static_size =
        is_dynamic ? std::dynamic_extent : static_sub_size;

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

    consteval static bool is_dynamic_extent(rank_type i) {
        return extents_type::static_extent(i) == std::dynamic_extent;
    }

    template <index_type... Indices>
    constexpr static extents_type convert_from(
        const zipper::extents<Indices...>& o) {
        if constexpr (sizeof...(Indices) == 0 && is_static) {
            return {};
        } else {
            return utils::extents::convert_extents<Extents>(o);
        }
    }

    template <concepts::ExtentsType Ext>
    constexpr static bool is_convertable_from() {
        return utils::extents::assignable_extents_v<Ext, extents_type>;
    }
    template <concepts::ExtentsType Ext>
    constexpr static bool is_convertable_from(const Ext& o) {
        return utils::extents::assignable_extents<Ext, extents_type>::value_runtime(o);
    }

    template <concepts::ExtentsType Ext>
    constexpr static bool is_resizeable_from() {
        return (Ext::rank() == 0 && is_static) || is_convertable_from<Ext>();
    }
    template <concepts::ExtentsType Ext>
    constexpr static bool is_resizeable_from(const Ext& o) {
        return (Ext::rank() == 0 && is_static) || is_convertable_from(o);
    }

    using dynamic_indices_helper = extents::DynamicExtentIndices<extents_type>;

    constexpr static std::array<rank_type, rank_dynamic> dynamic_indices =
        dynamic_indices_helper::value;
};
}  // namespace zipper::detail
#endif
