#if !defined(UVL_VIEWS_NULLARY_DETAIL_ALL_DYNAMIC_EXTENTS_HPP)
#define UVL_VIEWS_NULLARY_DETAIL_ALL_DYNAMIC_EXTENTS_HPP
#include "uvl/types.hpp"

namespace uvl::views::nullary {

template <index_type... Indices>
constexpr auto all_dynamic_extents() -> extents<Indices...> {
    using R = extents<Indices...>;
    constexpr static rank_type rank = R::rank_dynamic();
    std::array<index_type, rank> r;
    std::ranges::fill(r.begin(), r.end(), std::dynamic_extent);
}
}  // namespace uvl::views::nullary

#endif
