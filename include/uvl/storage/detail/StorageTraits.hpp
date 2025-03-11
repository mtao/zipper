#if !defined(UVL_STORAGE_DETAIL_STORAGE_TRAITS_HPP)
#define UVL_STORAGE_DETAIL_STORAGE_TRAITS_HPP
#include <mdspan/mdspan.hpp>

namespace uvl::storage::detail {
template <typename Derived_, typename... Extents>
struct StorageTraits {};
}  // namespace uvl::storage::detail
#endif
