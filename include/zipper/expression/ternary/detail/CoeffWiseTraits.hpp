#if !defined(ZIPPER_EXPRESSION_TERNARY_DETAIL_COEFFWISETRAITS_HPP)
#define ZIPPER_EXPRESSION_TERNARY_DETAIL_COEFFWISETRAITS_HPP

#include "zipper/expression/binary/detail/CoeffWiseTraits.hpp"

namespace zipper::expression::ternary::detail {

/// Merge extents from three children coefficient-wise.
/// Delegates pairwise to the binary merge utility, which handles
/// static-beats-dynamic and rank-0 (infinite/scalar) broadcasting.
template <typename EA, typename EB, typename EC>
struct coeffwise_extents_3 {
    using merged_ab = typename binary::detail::
        coeffwise_extents_values<EA, EB>::merged_extents_type;
    using merged_extents_type = typename binary::detail::
        coeffwise_extents_values<merged_ab, EC>::merged_extents_type;
};

} // namespace zipper::expression::ternary::detail
#endif
