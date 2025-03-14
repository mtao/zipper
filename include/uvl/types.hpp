#if !defined(UVL_DETAIL_TYPES_HPP)
#define UVL_DETAIL_TYPES_HPP
#include <cstddef>
#include <experimental/mdspan>

namespace uvl {
struct empty {};
using index_type = std::size_t;
using rank_type = std::size_t;
template <index_type... Extents>
using extents = std::experimental::extents<index_type, Extents...>;
template <rank_type N>
using dextents = std::experimental::dextents<index_type, N>;

template <typename... Args>
auto create_dextents(Args&&... args) {
    return dextents<sizeof...(Args)>(args...);
}

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
