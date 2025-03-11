#if !defined(UVL_STORAGE_STATICDENSESTORAGE_HPP)
#define UVL_STORAGE_STATICDENSESTORAGE_HPP

#include <array>

#include "PlainObjectStorageBase.hpp"
#include "uvl/types.hpp"

namespace uvl::storage {

template <typename ValueType, typename Extents,
          typename LayoutPolicy = uvl::default_layout_policy,
          typename AccessorPolicy = uvl::default_accessor_policy<ValueType>>
using StaticPlainObjectStorage = PlainObjectStorage<
    StaticPlainObjectValueAccessor<
        ValueType, uvl::detail::template ExtentsTraits<Extents>::static_size>,
    Extents, LayoutPolicy, AccessorPolicy>;

}  // namespace uvl::storage

/*
public:
using view_type = using view_type = views::PlainObjectStaticView<
    StaticPlainObjectStorage<ValueType, Extents, LayoutPolicy, AccessorPolicy>,
    Extents, LayoutPolicy, AccessorPolicy>;

using view_type::operator();
using view_type::coeff;
using view_type::coeff_ref;
};

}  // namespace uvl::storage
namespace uvl::views {

template <typename ValueType, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
struct detail::ViewTraits<uvl::storage::StaticPlainObjectStorage<
    ValueType, Extents, LayoutPolicy, AccessorPolicy>>
    : public detail::ViewTraits<views::PlainObjectStaticView<
          uvl::storage::StaticPlainObjectStorage<ValueType, Extents,
                                                 LayoutPolicy, AccessorPolicy>,
          uvl::storage::StaticPlainObjectValueAccessor<
              ValueType,
              uvl::detail::template ExtentsTraits<Extents>::static_size>,
          Extents, LayoutPolicy, AccessorPolicy>> {};
}  // namespace uvl::views
   */
#endif
