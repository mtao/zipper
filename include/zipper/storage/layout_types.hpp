#if !defined(ZIPPER_STORAGE_DETAIL_LAYOUT_TYPES_HPP)
#define ZIPPER_STORAGE_DETAIL_LAYOUT_TYPES_HPP

// grab mdspan / layouts
#include "zipper/types.hpp"

namespace zipper::storage {
using layout_right = std::experimental::layout_right;
using layout_left = std::experimental::layout_left;

// row major ~ layout_right
using row_major = layout_right;
using col_major = layout_left;

using default_layout_policy = layout_right;

template <bool IsRowMajor>
using matrix_layout = std::conditional_t<IsRowMajor, row_major, col_major>;

// alias for simply selecting layout
template <bool IsLeftMajor>
using tensor_layout =
    std::conditional_t<IsLeftMajor, layout_right, layout_left>;

static_assert(std::is_same_v<matrix_layout<false>, tensor_layout<false>>);
static_assert(std::is_same_v<matrix_layout<true>, tensor_layout<true>>);
}  // namespace zipper::storage

#endif
