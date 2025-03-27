
#if !defined(UVL_DETAIL_EXTENTS_INDEXED_STL_EXTENT_HPP)
#define UVL_DETAIL_EXTENTS_INDEXED_STL_EXTENT_HPP

// #include <array>
#include <experimental/mdspan>
// #include <map>
// #include <vector>

#include "uvl/concepts/IndexList.hpp"

namespace std::experimental::detail {

template <uvl::concepts::IndexList L>
inline constexpr integral_constant<size_t, 0> first_of(const L &s) {
    return integral_constant<size_t, 0>();
}

template <size_t k, class Extents, uvl::concepts::IndexList L>
inline constexpr size_t last_of(integral_constant<size_t, k>, const Extents &,
                                const L &i) {
    return i.size();
}

}  // namespace std::experimental::detail
#endif
