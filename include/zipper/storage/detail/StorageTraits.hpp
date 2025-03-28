#if !defined(ZIPPER_STORAGE_DETAIL_STORAGE_TRAITS_HPP)
#define ZIPPER_STORAGE_DETAIL_STORAGE_TRAITS_HPP
#include "zipper/types.hpp"

namespace zipper::storage::detail {
template <typename T>
struct StorageTraits;
//{
//    using value_type = ...;
//    using extents_type = ...;
//    using layout_policy = default_layout_policy;
//    using accessor_policy = default_accessor_policy<T>;
//};

template <typename Derived>
using mdspan_type =
    zipper::mdspan<typename StorageTraits<Derived>::value_type,
                typename StorageTraits<Derived>::Extents,
                typename StorageTraits<Derived>::layout_policy,
                typename StorageTraits<Derived>::accessor_policy>;
}  // namespace zipper::storage::detail
#endif
