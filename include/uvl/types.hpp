#if !defined(UVL_DETAIL_TYPES_HPP)
#define UVL_DETAIL_TYPES_HPP
#include <cstddef>
#include <experimental/mdspan>

#include "concepts/IndexLike.hpp"

namespace uvl {
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

template <concepts::IndexLike OffsetType = index_type,
          concepts::IndexLike ExtentType = index_type,
          concepts::IndexLike StrideType =
              std::integral_constant<index_type, 1>>
using slice_type =
    std::experimental::strided_slice<OffsetType, ExtentType, StrideType>;

template <concepts::IndexLike OffsetType, concepts::IndexLike ExtentType,
          concepts::IndexLike StrideType>
using slice_t = slice_type<OffsetType, ExtentType, StrideType>;

using default_layout_policy = std::experimental::layout_right;
template <typename T>
using default_accessor_policy = std::experimental::default_accessor<T>;

template <typename T, typename Extents,
          typename LayoutPolicy = default_layout_policy,
          typename AccessorPolicy = default_accessor_policy<T>>
using mdspan =
    std::experimental::mdspan<T, Extents, LayoutPolicy, AccessorPolicy>;
}  // namespace uvl
#endif
