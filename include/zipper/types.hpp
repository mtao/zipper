#if !defined(ZIPPER_DETAIL_TYPES_HPP)
#define ZIPPER_DETAIL_TYPES_HPP
#include <cstddef>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wextra-semi"
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wextra-semi"
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wunsafe-buffer-usage"
#else
#endif
#if defined(__cpp_lib_mdspan)
#include <mdspan>
#else
#include <mdspan/mdspan.hpp>
#endif
#pragma GCC diagnostic pop

#include "concepts/Index.hpp"

namespace zipper {
struct empty {};
using index_type = std::size_t;
using rank_type = std::size_t;
constexpr static index_type dynamic_extent = std::dynamic_extent;
template <index_type... Extents>
#if defined(__cpp_lib_mdspan)
using extents = std::extents<index_type, Extents...>;
// fully dynamic extents
template <rank_type N> using dextents = std::dextents<index_type, N>;
struct full_extent_t {};
template <typename OffsetType, typename ExtentType, typename StrideType>
struct strided_slice {
  OffsetType offset;
  ExtentType extent;
  StrideType stride;
};
template <typename OffsetType, typename ExtentType, typename StrideType>
using slice_type = strided_slice<OffsetType, ExtentType, StrideType>;
using default_layout_policy = std::layout_right;
template <typename T> using default_accessor_policy = std::default_accessor<T>;

template <typename T, typename Extents,
          typename LayoutPolicy = default_layout_policy,
          typename AccessorPolicy = default_accessor_policy<T>>
using mdspan = std::mdspan<T, Extents, LayoutPolicy, AccessorPolicy>;
#else
using extents = MDSPAN_IMPL_STANDARD_NAMESPACE::extents<index_type, Extents...>;
// fully dynamic extents
template <rank_type N>
using dextents = MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<index_type, N>;
using full_extent_t = MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent_t;
using full_extent_type = MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent_t;

template <typename OffsetType, typename ExtentType, typename StrideType>
using strided_slice =
    MDSPAN_IMPL_STANDARD_NAMESPACE::strided_slice<OffsetType, ExtentType,
                                                  StrideType>;

template <typename OffsetType, typename ExtentType, typename StrideType>
using slice_type = strided_slice<OffsetType, ExtentType, StrideType>;
using default_layout_policy = MDSPAN_IMPL_STANDARD_NAMESPACE::layout_right;
template <typename T>
using default_accessor_policy =
    MDSPAN_IMPL_STANDARD_NAMESPACE::default_accessor<T>;

template <typename T, typename Extents,
          typename LayoutPolicy = default_layout_policy,
          typename AccessorPolicy = default_accessor_policy<T>>
using mdspan = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<T, Extents, LayoutPolicy,
                                                      AccessorPolicy>;
#endif

// helper so users can pass in arguments to get extents
template <concepts::Index... Args> auto create_dextents(const Args &...args) {
  return dextents<sizeof...(Args)>(args...);
}

template <typename T, T v> struct constant {
  constexpr static T value = v;
  constexpr operator T() const { return value; }
  constexpr auto operator*() const -> T { return value; }
};

template <index_type N>
using static_index_t = std::integral_constant<index_type, N>;

template <rank_type N>
using static_rank_t = std::integral_constant<rank_type, N>;

// template <index_type N>
// constexpr static static_index_t<N> static_index = {};
//
// template <rank_type N>
// constexpr static static_rank_t<N> static_rank = {};

template <typename OffsetType, typename ExtentType, typename StrideType>
using slice_t = slice_type<OffsetType, ExtentType, StrideType>;

template <typename OffsetType = static_index_t<0>,
          typename ExtentType = static_index_t<dynamic_extent>,
          typename StrideType = static_index_t<1>>
inline auto slice(OffsetType start = {}, ExtentType size = {},
                  StrideType stride = {}) {
  static_assert(concepts::Index<OffsetType>);
  using OT = std::conditional_t<std::is_integral_v<std::decay_t<OffsetType>>,
                                index_type, std::decay_t<OffsetType>>;
  static_assert(concepts::Index<OT>);
  using ET = std::conditional_t<std::is_integral_v<std::decay_t<ExtentType>>,
                                index_type, std::decay_t<ExtentType>>;
  static_assert(concepts::Index<ET>);
  using ST = std::conditional_t<std::is_integral_v<OffsetType>, index_type,
                                std::decay_t<StrideType>>;
  static_assert(concepts::Index<ST>);
  return slice_t<OT, ET, ST>{OT(start), ET(size), ST(stride)};
}

template <index_type Offset, index_type Extent, index_type Stride = 1>
using static_slice_t = slice_t<std::integral_constant<index_type, Offset>,
                               std::integral_constant<index_type, Extent>,
                               std::integral_constant<index_type, Stride>>;

template <index_type Offset, index_type Extent, index_type Stride = 1>
inline auto static_slice() {
  return static_slice_t<Offset, Extent, Stride>{};
}

} // namespace zipper
#endif
