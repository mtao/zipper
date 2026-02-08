#if !defined(ZIPPER_UTILS_EXTENTS_IS_COMPATIBLE_HPP)
#define ZIPPER_UTILS_EXTENTS_IS_COMPATIBLE_HPP
#include "zipper/types.hpp"

#include "zipper/concepts/Extents.hpp"
#include "zipper/detail/ExtentsTraits.hpp"
#include "zipper/detail/pack_index.hpp"

namespace zipper::utils::extents {
namespace detail {
template <bool AllowDynamic, rank_type N, zipper::concepts::Extents Ext,
          index_type... Sizes>
constexpr bool is_dim_compatible(std::integer_sequence<index_type, Sizes...>) {
    constexpr static index_type R = zipper::detail::pack_index<N>(Sizes...);
    using traits = typename zipper::detail::ExtentsTraits<Ext>;
    if constexpr (AllowDynamic) {
        return R == Ext::static_extent(N) || R == std::dynamic_extent ||
               Ext::static_extent(N) == std::dynamic_extent;
    } else {
        static_assert(!traits::is_dynamic_extent(N));
        return R == Ext::static_extent(N);
    }
}
template <rank_type N, zipper::concepts::Extents Ext, index_type... Sizes>
bool is_dim_compatible(const Ext& ext,
                       std::integer_sequence<index_type, Sizes...>) {
    constexpr static index_type R = zipper::detail::pack_index<N>(Sizes...);
    using traits = typename zipper::detail::ExtentsTraits<Ext>;
    if constexpr (traits::is_dynamic_extent(N)) {
        return R == ext.extent(N);

    } else {
        return R == Ext::static_extent(N);
    }
}
template <zipper::concepts::Extents Ext, index_type... Indices,
          rank_type... N>
bool is_compatible(const Ext& ext,
                   std::integer_sequence<index_type, Indices...> idxs,
                   std::integer_sequence<rank_type, N...>) {
    return (is_dim_compatible<N>(ext, idxs) && ...);
}
template <bool AllowDynamic, zipper::concepts::Extents Ext,
          index_type... Indices, rank_type... N>
constexpr bool is_compatible(std::integer_sequence<index_type, Indices...> idxs,
                             std::integer_sequence<rank_type, N...>) {
    return (is_dim_compatible<AllowDynamic, N, Ext>(idxs) && ...);
}

}  // namespace detail

template <index_type... Indices, zipper::concepts::Extents Ext>
    requires(sizeof...(Indices) == Ext::rank() && Ext::rank_dynamic() == 0)
constexpr bool is_compatible() {
    return detail::is_compatible<false, Ext>(
        std::integer_sequence<index_type, Indices...>{},
        std::make_integer_sequence<rank_type, Ext::rank()>{});
}

template <index_type... Indices, zipper::concepts::Extents Ext>
    requires(sizeof...(Indices) == Ext::rank() && Ext::rank_dynamic() == 0)
constexpr bool is_compatible(const Ext&) {
    return detail::is_compatible<false, Ext>(
        std::integer_sequence<index_type, Indices...>{},
        std::make_integer_sequence<rank_type, Ext::rank()>{});
}

template <index_type... Indices, zipper::concepts::Extents Ext>
    requires(sizeof...(Indices) == Ext::rank() && Ext::rank_dynamic() != 0)
bool is_compatible(const Ext& ext) {
    return detail::is_compatible(
        ext, std::integer_sequence<index_type, Indices...>{},
        std::make_integer_sequence<rank_type, Ext::rank()>{});
}

template <index_type... Indices, zipper::concepts::Extents Ext>
    requires(sizeof...(Indices) == Ext::rank())
constexpr void throw_if_not_compatible(const Ext& ext) {
    if constexpr (Ext::rank_dynamic() == 0) {
        static_assert(
            Ext::rank_dynamic() == 0 ||
            detail::is_compatible<false, Ext>(
                std::integer_sequence<index_type, Indices...>{},
                std::make_integer_sequence<rank_type, Ext::rank()>{}));
    } else {
        if (!is_compatible<Indices...>(ext)) {
            throw std::invalid_argument("invalid extents");
            //    "Invalid extents {}, expected {}", ext,
            //    fmt::join(std::make_tuple(Indices...), ","));
        }
    }
}
}  // namespace zipper::utils::extents

#endif
