#if !defined(ZIPPER_STORAGE_DENSEACCESSORTYPES_HPP)
#define ZIPPER_STORAGE_DENSEACCESSORTYPES_HPP

#include "DenseAccessor.hpp"
#include "DenseData.hpp"

namespace zipper::storage {

template <typename T, typename Extents, typename LayoutPolicy = default_layout_policy,
          typename AccessorPolicy = zipper::default_accessor_policy<T>>
    using DenseAccesorFromType = DenseAccesor<DenseData<T,zipper::detail::ExtentsTraits<Extents>::static_size>, Extents, LayoutPolicy, AccessorPolicy>;
}

#endif
