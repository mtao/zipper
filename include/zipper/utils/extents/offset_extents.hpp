#if !defined(ZIPPER_UTILS_EXTENTS_OFFSET_EXTENTS_HPP)
#define ZIPPER_UTILS_EXTENTS_OFFSET_EXTENTS_HPP

#include "extent_arithmetic.hpp"
#include "zipper/concepts/ExtentsType.hpp"
#include "zipper/detail/extents/dynamic_extents_indices.hpp"
#include "zipper/detail/pack_index.hpp"
#include "zipper/types.hpp"

namespace zipper::utils::extents {

namespace detail {}
template <typename, typename>
struct OffsetExtents;
template <>
struct OffsetExtents<double, double> {
    // using extents_type = zipper::extents<offset(3, 1)>;
    using extents_type = double;
};

// template <>
// struct OffsetExtents<zipper::extents<3, 4>, zipper::signed_extents<0, 1>> {
//     using extents_type = double;
// };
// template <>
// struct OffsetExtents<zipper::extents<3>, zipper::signed_extents<1>> {
//    // using extents_type = zipper::extents<offset(3, 1)>;
//    using extents_type = double;
//};
//
// template <index_type FromIndices, int64_t OffsetIndices>
// struct OffsetExtents<zipper::extents<FromIndices>,
//                     zipper::signed_extents<OffsetIndices>> {
//    using extents_type = zipper::extents<offset(FromIndices, OffsetIndices)>;
//};
//
// template <index_type... FromIndices, int64_t... OffsetIndices>
// struct OffsetExtents<zipper::extents<FromIndices...>,
//                      zipper::extents<OffsetIndices...>> {
//     using extents_type = zipper::extents<FromIndices...>;
// };

template <index_type... FromIndices, int64_t... OffsetIndices>
struct OffsetExtents<zipper::extents<FromIndices...>,
                     std::integer_sequence<int64_t, OffsetIndices...>> {
    using extents_type = zipper::extents<offset(FromIndices, OffsetIndices)...>;

    template <rank_type... N>
    static extents_type run_dynamic(const zipper::extents<FromIndices...>& s,
                                    std::integer_sequence<rank_type, N...>) {
        constexpr rank_type my_dynamic_rank = sizeof...(N);
        using FE = zipper::extents<FromIndices...>;
        constexpr auto dyn_indices =
            zipper::detail::extents::dynamic_extents_indices_v<FE>;
        std::array<index_type, my_dynamic_rank> arr = {{offset(
            s.extent(dyn_indices[N]),
            zipper::detail::pack_index<dyn_indices[N]>(OffsetIndices...))...}};
        return extents_type(arr);
    }  // namespace zipper::utils::extents

    static extents_type run(const zipper::extents<FromIndices...>& s)
        requires(sizeof...(FromIndices) == sizeof...(OffsetIndices))
    {
        constexpr static rank_type my_dynamic_rank =
            extents_type ::rank_dynamic();

        if constexpr (my_dynamic_rank == 0) {
            return extents_type{};
        } else {
            return run_dynamic(
                s, std::make_integer_sequence<rank_type, my_dynamic_rank>{});
        }
    }
};

template <concepts::ExtentsType A, int64_t... OffsetIndices>
using offset_extents_t = typename OffsetExtents<
    A, std::integer_sequence<int64_t, OffsetIndices...>>::extents_type;

template <int64_t... OffsetIndices, concepts::ExtentsType A>
auto offset_extents(const A& a) {
    return OffsetExtents<
        A, std::integer_sequence<int64_t, OffsetIndices...>>::run(a);
}

// template <index_type... OffsetIndices, index_type... FromIndices>
// auto merge_extents(const extents<OffsetIndices...>& a,
// const extents<FromIndices...>& b
//         ) {
//     return OffsetExtents<extents<OffsetIndices...>, A>::run(a);
// }
}  // namespace zipper::utils::extents
#endif
