#if !defined(ZIPPER_DETAIL_EXTENTS_SLICE_EXTENTS_HPP)
#define ZIPPER_DETAIL_EXTENTS_SLICE_EXTENTS_HPP

#include <type_traits>

#include "zipper/concepts/Extents.hpp"
#include "zipper/concepts/Index.hpp"
#include "zipper/concepts/SliceLike.hpp"
#include "zipper/types.hpp"

namespace zipper::detail::extents {
// slices that have an extent of std::dynamic_extent
template <concepts::Extents Extents>
struct SliceSpecialization {
    template <rank_type J, concepts::IndexLike Idx, concepts::SliceLike Slice>
    auto extent(const Extents& ext, Slice const& s) {
        if constexpr (concepts::IndexLike<Slice> ||
                      std::is_same_v<Slice, zipper::full_extent_type>) {
            return s;
        } else {
            constexpr auto extent = s.extent;
            if constexpr (Extents::static_extent(J) == std::dynamic_extent) {
                if (extent == std::dynamic_extent) {
                    return ext.extent(J);
                } else {
                    return extent;
                }
            } else if constexpr (extent == std::dynamic_extent) {
                return ext.extent(J);
            } else {
                return extent;
            }
        }
    }

    template <concepts::SliceLike... Slices>
    static auto update_extents(const Extents& ext, const Slices&... s) {}
    template <typename Index>
    auto transform_slice_offset(const Index& s) {
        return s;
    }
    template <rank_type Dim, concepts::SliceLike Slice>
    auto transform_slice_extent(const Extents& e, const Slice& s) {
        if constexpr (std::is_integral_constant_v<Index>) {
            if constexpr (Index::value == std::dynamic_extent) {
                return extent(Dim);
            } else {
                return s;
            }
        } else if constexpr (std::is_integral_v<SL>) {
            return
        }
    }

    template <concepts::SliceLike Slice, rank_type Dim>
    auto transform_slice(const Slice& s) {
        using SL = std::decay_t<Slice>;
        if constexpr (zipper::detail::is_integral_constant_v<SL>) {
            return s;
        } else if constexpr (std::is_integral_v<SL>) {
            return s;
        } else if constexpr (std::is_same_v<SL, zipper::full_extent_type>) {
            return s;
        } else {
            const auto offset = std::experimental::detail::first_of(s);
            const auto stride = std::experimental::detail::stride_of(s);
            const auto extent = std::experimental::detail::stride_of(s);

            using O = std::decay_t<decltype(offset)>;
            using E = std::decay_t<decltype(extent)>;
            using S = std::decay_t<decltype(stride)>;

            zipper::detail::is_integral_constant_v<O>;

            return zipper::slice_t<O, E, S>(offset, e, s);
        }
    }
};

}  // namespace zipper::detail::extents
#endif
