#if !defined(UVL_STORAGE_DETAIL_VIEW_TRAITS_HPP)
#define UVL_STORAGE_DETAIL_VIEW_TRAITS_HPP
#include "uvl/types.hpp"

namespace uvl::storage::detail {
template <typename T>
struct ViewTraits;
//{
//    using value_type = ...;
//    using extents_type = ...;
//    using layout_policy = default_layout_policy;
//    using accessor_policy = default_accessor_policy<T>;
//};

template <typename Derived>
using mdspan_type =
    uvl::mdspan<typename ViewTraits<Derived>::value_type,
                typename ViewTraits<Derived>::Extents,
                typename ViewTraits<Derived>::layout_policy,
                typename ViewTraits<Derived>::accessor_policy>;
}  // namespace uvl::storage::detail
#endif
