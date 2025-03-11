
#if !defined(UVL_VIEWS_BINARY_DETAIL_COEFFWISETRAITS_HPP)
#define UVL_VIEWS_BINARY_DETAIL_COEFFWISETRAITS_HPP
#include "uvl/views/detail/ViewTraits.hpp"

namespace uvl::views::binary::detail {

template <typename, typename>
struct coeffwise_extents_values;
template <index_type... Idxs, index_type... Idxs2>
struct coeffwise_extents_values<extents<Idxs...>, extents<Idxs2...>> {
    using merged_extents_type =
        extents<(Idxs == std::dynamic_extent ? Idxs2 : Idxs)...>;
};

template <typename A, typename B>
struct CoeffWiseTraits {
    using LHSTraits = uvl::views::detail::ViewTraits<A>;
    using RHSTraits = uvl::views::detail::ViewTraits<B>;
    // static_assert(LHSTraits::value_type == RHSTraits::value_type);
    using value_type = typename LHSTraits::value_type;
    using extents_type = typename LHSTraits::extents_type;
    // coeffwise_extents_values<typename LHSTraits::extents_type, typename
    // RHSTraits::extents_type>::merged_extents_type;
};
}  // namespace uvl::views::binary::detail
#endif
