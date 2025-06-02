
#if !defined(ZIPPER_DETAIL_EXTENTS_CONSTEXPR_EXTENTS_HPP)
#define ZIPPER_DETAIL_EXTENTS_CONSTEXPR_EXTENTS_HPP

#include "zipper/types.hpp"
namespace zipper::detail::extents {

template <rank_type N, index_type... I>
auto constexpr_extent(const zipper::extents<I...>& i) {
    using ET = zipper::extents<I...>;
    constexpr static index_type S = ET::static_extent(N);
    if constexpr (S == std::dynamic_extent) {
        return index_type(i.extent(N));
    } else {
        return std::integral_constant<index_type, S>{};
    }
}
}  // namespace zipper::detail::extents

#endif
