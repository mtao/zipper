#if !defined(UVL_DATA_STATICDENSESTORAGE_HPP)
#define UVL_DATA_STATICDENSESTORAGE_HPP

#include <array>

#include "StaticView.hpp"
#include "uvl/types.hpp"

namespace uvl::storage {

template <typename ValueType, std::size_t N>
class StaticPlainObjectValueAccessor {
   public:
    using value_type = ValueType;
    constexpr static auto size() -> std::size_t { return N; }
    value_type coeff(index_type i) const { return m_data[i]; }
    value_type& coeff_ref(index_type i) { return m_data[i]; }
    const value_type& const_coeff_ref(index_type i) const { return m_data[i]; }
    value_type* data() { return m_data.data(); }
    const value_type* data() const { return m_data.data(); }

   private:
    std::array<value_type, N> m_data;
};

template <typename ValueType, typename Extents,
          typename LayoutPolicy = uvl::default_layout_policy,
          typename AccessorPolicy = uvl::default_accessor_policy<ValueType>>
class StaticPlainObjectStorage
    : public StaticView<StaticPlainObjectValueAccessor<
                            ValueType, uvl::detail::template ExtentsTraits<
                                           Extents>::static_size>,
                        Extents, LayoutPolicy, AccessorPolicy> {};

}  // namespace uvl::storage
#endif
