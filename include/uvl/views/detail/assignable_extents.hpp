#include "uvl/concepts/TensorViewDerived.hpp"
#include "uvl/concepts/ViewDerived.hpp"
#if !defined(UVL_VIEWS_DETAIL_ASSIGNABLE_EXTENTS_HPP)
#define UVL_VIEWS_DETAIL_ASSIGNABLE_EXTENTS_HPP
#include "uvl/concepts/ViewDerived.hpp"
#include "uvl/types.hpp"
namespace uvl::views::detail {
template <typename, typename>
struct assignable_extents;
template <index_type... Idxs, index_type... Idxs2>
struct assignable_extents<extents<Idxs...>, extents<Idxs2...>> {
    constexpr static bool equal_sizes = sizeof...(Idxs) == sizeof...(Idxs2);
    constexpr static bool dynamic_matches =
        ((Idxs != std::dynamic_extent && Idxs2 != std::dynamic_extent
              ? Idxs == Idxs2
              : true) &&
         ...);
    constexpr static bool value = equal_sizes && dynamic_matches;
};
// template <concepts::ViewDerived A, concepts::ViewDerived B>
// struct assignable_extents<A, B> {
//     using ATraits = ViewTraits<A>;
//     using BTraits = ViewTraits<B>;
//     constexpr static bool value =
//         assignable_extents<typename ATraits::extents_type,
//                            typename BTraits::extents_type>::value;
// };
}  // namespace uvl::views::detail
#endif
