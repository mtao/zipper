
#if !defined(ZIPPER_UTILS_EXTENTS_IS_CUBIC_HPP)
#define ZIPPER_UTILS_EXTENTS_IS_CUBIC_HPP
#include "zipper/concepts/ExtentsType.hpp"
#include "zipper/detail/ExtentsTraits.hpp"
#include "zipper/detail/pack_index.hpp"
#include "zipper/types.hpp"

namespace zipper::utils::extents {
namespace detail {
template <zipper::concepts::ExtentsType Ext, rank_type... N>
constexpr index_type max_dim(std::integer_sequence<rank_type, N...>) {
    static_assert(Ext::rank_dynamic() == 0);
    return std::max({(Ext::static_extent(N) == std::dynamic_extent
                          ? 0
                          : Ext::static_extent(N))...});
    return std::max({Ext::static_extent(N)...});
}
template <zipper::concepts::ExtentsType Ext>
constexpr index_type max_dim() {
    return max_dim<Ext>(std::make_integer_sequence<rank_type, Ext::rank()>{});
}

template <zipper::concepts::ExtentsType Ext, rank_type... N>
index_type max_dim(const Ext& e, std::integer_sequence<rank_type, N...>) {
    return std::max({e.extent(N)...});
}
template <zipper::concepts::ExtentsType Ext>
index_type max_dim(const Ext& e) {
    return max_dim<Ext>(e,
                        std::make_integer_sequence<rank_type, Ext::rank()>{});
}
template <zipper::concepts::ExtentsType Ext, rank_type... N>
bool is_cubic(const Ext& ext, std::integer_sequence<rank_type, N...> n) {
    const index_type max = max_dim<Ext>(ext, n);
    return ((ext.extent(N) == max) && ...);
}
template <zipper::concepts::ExtentsType Ext, rank_type... N>
constexpr bool is_cubic(std::integer_sequence<rank_type, N...> n) {
    constexpr static index_type max = max_dim<Ext>(n);
    return ((Ext::static_extent(N) == std::dynamic_extent ||
             Ext::static_extent(N) == max) &&
            ...);
}

}  // namespace detail
//
//
/*
    template <zipper::concepts::ExtentsType Ext,zipper::concepts::ExtentsType
   Ext2> requires(Ext2::rank() == Ext::rank()) constexpr bool
   is_cubic(const Ext& target_ext, const Ext2& ext) { using traits =
   typename zipper::detail::ExtentsTraits<Ext>;

            return traits::is_convertable_from(ext);


        }
*/
template <zipper::concepts::ExtentsType Ext>
constexpr bool is_cubic() {
    return detail::is_cubic<Ext>(
        std::make_integer_sequence<rank_type, Ext::rank()>{});
}

template <zipper::concepts::ExtentsType Ext>
    requires(Ext::rank_dynamic() == 0)
constexpr bool is_cubic(const Ext&) {
    return detail::is_cubic<Ext>(
        std::make_integer_sequence<rank_type, Ext::rank()>{});
}

template <zipper::concepts::ExtentsType Ext>
    requires(Ext::rank_dynamic() != 0)
bool is_cubic(const Ext& ext) {
    return detail::is_cubic(
        ext, std::make_integer_sequence<rank_type, Ext::rank()>{});
}

template <zipper::concepts::ExtentsType Ext>
constexpr void throw_if_not_cubic(const Ext& ext) {
    if constexpr (Ext::rank_dynamic() == 0) {
        static_assert(
            Ext::rank_dynamic() == 0 ||
            detail::is_cubic<Ext>(
                std::make_integer_sequence<rank_type, Ext::rank()>{}));
    } else {
        if (!is_cubic(ext)) {
            throw std::invalid_argument("invalid extents, expected cubic");
            //    "Invalid extents {}, expected {}", ext,
            //    fmt::join(std::make_tuple(Indices...), ","));
        }
    }
}
}  // namespace zipper::utils::extents

#endif
