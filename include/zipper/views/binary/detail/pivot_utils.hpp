#if !defined(ZIPPER_VIEWS_BINARY_DETAIL_PIVOT_UTILS_HPP)
#define ZIPPER_VIEWS_BINARY_DETAIL_PIVOT_UTILS_HPP
#include "zipper/detail//ExtentsTraits.hpp"

namespace zipper::views::binary::detail {
    template <rank_type a, rank_type b, index_type... Indices, rank_type... Idxs>
        auto pivot(const extents<Indices...>& indices , std::integer_sequence<rank_type,Idxs...>) {

            constexpr auto swap_ab = [](rank_type x) -> rank_type {
                if(x == a) {
                    return a;
                } else if(x == b) {
                    return b;
                } else {
                    return x;
                }
            };
            std::array<index_type, extents<Indices...>::rank_dynamic()> dyn;
            rank_type idx = 0;
            for(rank_type j = 0; j < dyn.size(); ++j) {

                }

            return extents<swap_ab(Indices)...>();



        }
    
    template <rank_type a, rank_type b, index_type... Indices>
        auto pivot(const extents<Indices...>& indices = {}) {
            return pivot(indices, std::make_integer_sequence<rank_type, sizeof...(Indices)>{});


        }


}


}
