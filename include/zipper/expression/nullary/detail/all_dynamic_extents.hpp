#if !defined(ZIPPER_VIEWS_NULLARY_DETAIL_ALL_DYNAMIC_EXTENTS_HPP)
#define ZIPPER_VIEWS_NULLARY_DETAIL_ALL_DYNAMIC_EXTENTS_HPP
#include "zipper/types.hpp"

namespace zipper::views::nullary {

template <index_type... Indices>
consteval auto all_dynamic_extents() -> extents<Indices...> {
    using R = extents<Indices...>;
    constexpr static rank_type rank = R::rank_dynamic();
    std::array<index_type, rank> r;
    std::ranges::fill(r.begin(), r.end(), std::dynamic_extent);
    return r;
}
}  // namespace zipper::views::nullary

#endif
