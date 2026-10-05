#if !defined(ZIPPER_STORAGE_LAYOUT_TRAITS_HPP)
#define ZIPPER_STORAGE_LAYOUT_TRAITS_HPP

/// @file layout_traits.hpp
/// @brief Compile-time properties of layout mapping types.
///
///   fastest_dimension_v<Mapping>   the dimension along which consecutive
///                                  indices are adjacent in memory (smallest
///                                  stride), or unknown_dimension when the
///                                  mapping type does not determine it (e.g.
///                                  layout_stride, whose strides are run-time).
///
/// It is a property of the layout alone: row-major layouts (layout_right,
/// layout_right_padded) run fastest along their last dimension, column-major
/// ones along their first, and layout_permuted along wherever its child's
/// fastest dimension went.

#include <cstddef>
#include <span>

#include "zipper/storage/layout_permuted.hpp"
#include "zipper/storage/layout_types.hpp"
#include "zipper/types.hpp"

namespace zipper::storage {

/// "Not determined by the type" for dimension-valued layout properties.
inline constexpr rank_type unknown_dimension = std::dynamic_extent;

namespace detail {
    /// Fastest dimension of a rank-`Rank` mapping of layout policy `Layout`.
    template <typename Layout, std::size_t Rank>
    struct layout_fastest_dimension {
        static constexpr rank_type value = unknown_dimension;
    };

    template <std::size_t Rank>
        requires(Rank > 0)
    struct layout_fastest_dimension<layout_right, Rank> {
        static constexpr rank_type value = Rank - 1;
    };
    template <std::size_t Rank>
        requires(Rank > 0)
    struct layout_fastest_dimension<layout_left, Rank> {
        static constexpr rank_type value = 0;
    };

#if !defined(__cpp_lib_mdspan)
    template <std::size_t Pad, std::size_t Rank>
        requires(Rank > 0)
    struct layout_fastest_dimension<
        MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE::
            layout_right_padded<Pad>,
        Rank> {
        static constexpr rank_type value = Rank - 1;
    };
    template <std::size_t Pad, std::size_t Rank>
        requires(Rank > 0)
    struct layout_fastest_dimension<
        MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE::
            layout_left_padded<Pad>,
        Rank> {
        static constexpr rank_type value = 0;
    };
#endif
} // namespace detail

/// Fastest dimension of a layout mapping type (see file docs).
template <typename Mapping>
inline constexpr rank_type fastest_dimension_v =
    detail::layout_fastest_dimension<typename Mapping::layout_type,
                                     Mapping::extents_type::rank()>::value;

namespace detail {
    template <typename ChildMapping, std::size_t... Perm>
    struct layout_fastest_dimension<layout_permuted<ChildMapping, Perm...>,
                                    sizeof...(Perm)> {
        static constexpr rank_type value = [] {
            constexpr rank_type child = fastest_dimension_v<ChildMapping>;
            constexpr std::array<std::size_t, sizeof...(Perm)> perm{{Perm...}};
            for (rank_type d = 0; d < perm.size(); ++d) {
                if (child != unknown_dimension && perm[d] == child) {
                    return d;
                }
            }
            return unknown_dimension;
        }();
    };
} // namespace detail

} // namespace zipper::storage

#endif
