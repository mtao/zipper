#if !defined(ZIPPER_DETAIL_EXTENTS_GET_EXTENT_HPP)
#define ZIPPER_DETAIL_EXTENTS_GET_EXTENT_HPP

#include "zipper/concepts/Extents.hpp"
#include "zipper/detail/constexpr_arithmetic.hpp"
namespace zipper::detail::extents {

template <rank_type N, concepts::Extents Extents>
constexpr auto get_extent(const Extents& e) {
    if constexpr (Extents::static_extent(N) == std::dynamic_extent) {
        return zipper::detail::ConstexprArithmetic(e.extent(N));
    } else {
        return zipper::detail::ConstexprArithmetic<
            static_index_t<Extents::static_extent(N)>>{};
    }
}

}  // namespace zipper::detail::extents
#endif
