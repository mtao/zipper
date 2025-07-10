#if !defined(ZIPPER_STORAGE_DETAIL_ACCESSOR_POLICIES_HPP)
#define ZIPPER_STORAGE_DETAIL_ACCESSOR_POLICIES_HPP

// grab mdspan / layouts
#include "zipper/types.hpp"

namespace zipper::storage {
template <typename T>
using default_accessor_policy = std::experimental::default_accessor<T>;
}  // namespace zipper::storage

#endif

