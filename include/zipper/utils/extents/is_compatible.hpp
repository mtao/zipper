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

// MSVC cannot deduce a template parameter that follows a parameter pack
// (C3547).  We restructure the public API so that the Extents type Ext is
// either deduced from a function parameter, or wrapped inside a helper that
// carries the pack as an integer_sequence instead of a bare pack after Ext.

/// Compile-time compatibility check (no runtime Ext argument).
/// Callers write: is_compatible<I0, I1, ...>(ext) — but Ext is deduced from
/// the function argument, not placed after the pack.
///
/// For the zero-argument overload we use a helper so that Ext appears before
/// the pack.
namespace detail {
template <zipper::concepts::Extents Ext, index_type... Indices>
    requires(sizeof...(Indices) == Ext::rank() && Ext::rank_dynamic() == 0)
constexpr bool is_compatible_static() {
    return is_compatible<false, Ext>(
        std::integer_sequence<index_type, Indices...>{},
        std::make_integer_sequence<rank_type, Ext::rank()>{});
}
}  // namespace detail

// The original public API used:
//   template <index_type... Indices, Extents Ext>
// MSVC rejects this (C3547).  Instead we provide overloads where Ext is
// deduced from a function parameter, and delegate the no-arg overload to
// a helper with reversed parameter order.

template <index_type... Indices, zipper::concepts::Extents Ext>
    requires(sizeof...(Indices) == Ext::rank() && Ext::rank_dynamic() == 0)
constexpr bool is_compatible(const Ext&) {
    return detail::is_compatible_static<Ext, Indices...>();
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
            detail::is_compatible_static<Ext, Indices...>());
    } else {
        if (!is_compatible<Indices...>(ext)) {
            throw std::invalid_argument("invalid extents");
        }
    }
}
}  // namespace zipper::utils::extents

#endif
