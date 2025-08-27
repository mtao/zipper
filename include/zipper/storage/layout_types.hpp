#if !defined(ZIPPER_STORAGE_DETAIL_LAYOUT_TYPES_HPP)
#define ZIPPER_STORAGE_DETAIL_LAYOUT_TYPES_HPP

// grab mdspan / layouts
#include "zipper/types.hpp"

namespace zipper::storage {
#if defined(__cpp_lib_mdspan)
using layout_right = std::layout_right;
using layout_left = std::layout_left;
#else
using layout_right = MDSPAN_IMPL_STANDARD_NAMESPACE::layout_right;
using layout_left = MDSPAN_IMPL_STANDARD_NAMESPACE::layout_left;
#endif

// row major ~ layout_right
using row_major = layout_right;
using col_major = layout_left;

// Default using layout_right i.e row major optimizes for striding in dense
// matrix-vector products
// That is, with row major we get 0,1,2 contiguous
// [ 0 1 2 ]   [ a ] 
// [ 3 4 5 ] x [ b ] 
// [ 6 7 8 ]   [ c ]
// so evaluating one row requires loading [0,1,2] and [a,b,c] into memory, which is 2 contiguous blocks
// On the other hand,
// [ 0 3 6 ]   [ a ] 
// [ 1 4 7 ] x [ b ] 
// [ 2 5 8 ]   [ c ]
// requires loading 0,3,6 which, for larger sizes, might not be contiguous
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
