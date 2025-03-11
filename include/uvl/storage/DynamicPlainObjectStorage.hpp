
#if !defined(UVL_STORAGE_DYNAMICDENSESTORAGE_HPP)
#define UVL_STORAGE_DYNAMICDENSESTORAGE_HPP

#include <vector>

#include "PlainObjectStorage.hpp"
#include "uvl/types.hpp"

namespace uvl::storage {

template <typename ValueType>
class DynamicPlainObjectValueAccessor {
   public:
    DynamicPlainObjectValueAccessor() = default;
    DynamicPlainObjectValueAccessor(index_type index) : m_data(index) {}
    using value_type = ValueType;
    auto size() const -> std::size_t { return m_data.size(); }
    value_type coeff(index_type i) const { return m_data[i]; }
    value_type& coeff_ref(index_type i) { return m_data[i]; }
    const value_type& const_coeff_ref(index_type i) const { return m_data[i]; }
    value_type* data() { return m_data.data(); }
    const value_type* data() const { return m_data.data(); }

   private:
    std::vector<value_type> m_data;
};

template <typename ValueType, typename Extents,
          typename LayoutPolicy = uvl::default_layout_policy,
          typename AccessorPolicy = uvl::default_accessor_policy<ValueType>>
class DynamicPlainObjectStorage
    : public views::PlainObjectViewBase<DynamicPlainObjectStorage<
          ValueType, Extents, LayoutPolicy, AccessorPolicy>>

      public : using view_type = views::PlainObjectDynamicView<
                   DynamicPlainObjectStorage<ValueType, Extents, LayoutPolicy,
                                             AccessorPolicy>,
                   DynamicPlainObjectValueAccessor<ValueType>, Extents,
                   LayoutPolicy, AccessorPolicy>;
using extents_type = view_type::extents_type;
DynamicPlainObjectStorage(const extents_type& extents)
    : view_type(extents, view_type::size_from_extents(extents)) {}

using view_type::operator();
};

}  // namespace uvl::storage
namespace uvl::views {

template <typename ValueType, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
struct detail::ViewTraits<uvl::storage::DynamicPlainObjectStorage<
    ValueType, Extents, LayoutPolicy, AccessorPolicy>>
    : public detail::ViewTraits<views::PlainObjectDynamicView<
          uvl::storage::DynamicPlainObjectStorage<ValueType, Extents,
                                                  LayoutPolicy, AccessorPolicy>,
          uvl::storage::DynamicPlainObjectValueAccessor<ValueType>, Extents,
          LayoutPolicy, AccessorPolicy>> {};
}  // namespace uvl::views
#endif
