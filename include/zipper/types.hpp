#if !defined(ZIPPER_DETAIL_TYPES_HPP)
#define ZIPPER_DETAIL_TYPES_HPP
#include <cstddef>
#include <experimental/mdspan>

#include "concepts/IndexLike.hpp"

namespace zipper {
struct empty {};
using index_type = std::size_t;
using rank_type = std::size_t;
template <index_type... Extents>
using extents = std::experimental::extents<index_type, Extents...>;

// fully dynamic extents
template <rank_type N>
using dextents = std::experimental::dextents<index_type, N>;

// helper so users can pass in arguments to get extents
template <concepts::IndexLike... Args>
auto create_dextents(const Args&... args) {
    return dextents<sizeof...(Args)>(args...);
}

using full_extent_t = std::experimental::full_extent_t;
using full_extent_type = std::experimental::full_extent_t;
constexpr static full_extent_type full_extent;

template <index_type N>
constexpr static std::integral_constant<index_type, N> static_index =
    std::integral_constant<index_type, N>{};

template <typename OffsetType, typename ExtentType, typename StrideType>
using slice_type =
    std::experimental::strided_slice<OffsetType, ExtentType, StrideType>;

template <typename OffsetType, typename ExtentType, typename StrideType>
using slice_t = slice_type<OffsetType, ExtentType, StrideType>;

template <typename OffsetType = std::integral_constant<index_type, 0>,
          typename ExtentType =
              std::integral_constant<index_type, std::dynamic_extent>,
          typename StrideType = std::integral_constant<index_type, 1>>
inline auto slice(OffsetType start = {}, ExtentType size = {},
                  StrideType stride = {}) {
    static_assert(concepts::IndexLike<OffsetType>);
    using OT = std::conditional_t<std::is_integral_v<std::decay_t<OffsetType>>,
                                  index_type, std::decay_t<OffsetType>>;
    static_assert(concepts::IndexLike<OT>);
    using ET = std::conditional_t<std::is_integral_v<std::decay_t<ExtentType>>,
                                  index_type, std::decay_t<ExtentType>>;
    static_assert(concepts::IndexLike<ET>);
    using ST = std::conditional_t<std::is_integral_v<OffsetType>, index_type,
                                  OffsetType>;
    static_assert(concepts::IndexLike<ST>);
    return slice_t<OT, ET, ST>{OT(start), ET(size), ST(stride)};
}

using default_layout_policy = std::experimental::layout_right;
template <typename T>
using default_accessor_policy = std::experimental::default_accessor<T>;

template <typename T, typename Extents,
          typename LayoutPolicy = default_layout_policy,
          typename AccessorPolicy = default_accessor_policy<T>>
using mdspan =
    std::experimental::mdspan<T, Extents, LayoutPolicy, AccessorPolicy>;
}  // namespace zipper
#endif
