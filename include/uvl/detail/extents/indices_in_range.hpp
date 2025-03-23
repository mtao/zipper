#

#if !defined(UVL_DETAIL_EXTENTS_INDICES_IN_RANGE_HPP)
#define UVL_DETAIL_EXTENTS_INDICES_IN_RANGE_HPP

#include "uvl/concepts/ExtentsType.hpp"
#include "uvl/concepts/IndexPackLike.hpp"
#include "uvl/concepts/SlicePackLike.hpp"
#include "uvl/concepts/TupleLike.hpp"
#include "uvl/concepts/ViewAccessPack.hpp"
#include "uvl/types.hpp"

namespace uvl::detail::extents {

template <concepts::ExtentsType Extents, typename... Indices, rank_type... N>
    requires(concepts::IndexPackLike<Indices...>)
bool indices_in_range(std::integer_sequence<rank_type, N...>, const Extents& e,
                      const Indices&... i) {
    auto check = []<rank_type J, typename T>(
                     std::integral_constant<rank_type, J>, const Extents& e,
                     const T& index) -> bool {
        static_assert(concepts::SliceLike<T>);
        if constexpr (concepts::IndexLike<T>) {
            if (index_type(index) < 0) {
                return false;
            }
            if constexpr (Extents::static_extent(J) == std::dynamic_extent) {
                return index_type(index) < e.extent(J);
            } else {
                return index_type(index) < Extents::static_extent(J);
            }
        } else if constexpr (std::is_same_v<full_extent_type,
                                            std::decay_t<T>>) {
            return true;
        } else {
            constexpr index_type start =
                std::experimental::detail::first_of(index);
            constexpr index_type end =
                std::experimental::detail::last_of(index);
            if (0 < start) {
                return false;
            }

            if constexpr (Extents::static_extent(J) == std::dynamic_extent) {
                return index_type(end) < e.extent(J);
            } else {
                return index_type(end) < Extents::static_extent(J);
            }
        }
    };

    return (check(std::integral_constant<rank_type, N>{}, e, i) && ...);
}

template <concepts::ExtentsType Extents, typename... Indices>
    requires(concepts::IndexPackLike<Indices...>)
bool indices_in_range(const Extents& e, const Indices&... i) {
    if constexpr (Extents::rank() == 0) {
        return true;
    } else {
        static_assert(Extents::rank() == sizeof...(Indices));
        return indices_in_range(
            std::make_integer_sequence<rank_type, Extents::rank()>{}, e, i...);
    }
}
}  // namespace uvl::detail::extents
#endif
