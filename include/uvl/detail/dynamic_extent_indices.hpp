
#define UVL_DETAIL_DYNAMIC_EXTENT_INDICES_HPP

#include "uvl/types.hpp"

namespace uvl::detail {

//template <index_type... Indices>
//constexpr auto dynamic_swizzle_extents(index_type... Indices) {
//    using E = extents<Indices...>;
//}
template <index_type... Indices>
constexpr auto dynamic_swizzle_extents(const extents<Indices...>) {

    using E = extents<Indices...>;
}

}  // namespace uvl::detail
#endif
