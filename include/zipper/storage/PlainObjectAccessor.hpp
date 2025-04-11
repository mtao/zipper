#if !defined(ZIPPER_STORAGE_PLAINOBJECTACCESSOR_HPP)
#define ZIPPER_STORAGE_PLAINOBJECTACCESSOR_HPP

#include <array>
#include <vector>

#include "zipper/detail//ExtentsTraits.hpp"
namespace zipper::storage {
template <typename ValueType, std::size_t N_>
class StaticValueAccessor {
    constexpr static std::size_t N = std::max<std::size_t>(
        N_, 1);  // if we have empty extents we have a scalar

   public:
    using value_type = ValueType;
    constexpr static auto size() -> std::size_t { return N; }
    value_type coeff(index_type i) const { return m_data[i]; }
    value_type& coeff_ref(index_type i) { return m_data[i]; }
    const value_type& const_coeff_ref(index_type i) const { return m_data[i]; }
    value_type* data() { return m_data.data(); }
    const value_type* data() const { return m_data.data(); }

    const auto& container() const { return m_data; }
    auto& container() { return m_data; }

   private:
    std::array<value_type, N> m_data;
};
template <typename ValueType>
class DynamicValueAccessor {
   public:
    DynamicValueAccessor() = default;
    DynamicValueAccessor(index_type index) : m_data(index) {}
    using value_type = ValueType;
    auto size() const -> std::size_t { return m_data.size(); }
    value_type coeff(index_type i) const { return m_data[i]; }
    value_type& coeff_ref(index_type i) { return m_data[i]; }
    const value_type& const_coeff_ref(index_type i) const { return m_data[i]; }
    value_type* data() { return m_data.data(); }
    const value_type* data() const { return m_data.data(); }

    const auto& container() const { return m_data; }
    auto& container() { return m_data; }

   private:
    std::vector<value_type> m_data;
};

template <typename ValueType, typename Extents>
using PlainObjectAccessor = std::conditional_t<
    zipper::detail::ExtentsTraits<Extents>::is_static,
    StaticValueAccessor<ValueType, zipper::detail::template ExtentsTraits<
                                       Extents>::static_size>,
    DynamicValueAccessor<ValueType>>;
}  // namespace zipper::storage
#endif
