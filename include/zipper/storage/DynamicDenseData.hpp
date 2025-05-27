
#if !defined(ZIPPER_STORAGE_DYNAMICDENSEDATA_HPP)
#define ZIPPER_STORAGE_DYNAMICDENSEDATA_HPP

#include <vector>
#include <zipper/types.hpp>

namespace zipper::storage {
template <typename ValueType>
class DynamicDenseData {
   public:
    DynamicDenseData() = default;
    DynamicDenseData(index_type size) : m_data(size) {}
    using value_type = ValueType;
    using storage_type = std::vector<value_type>;
    constexpr static index_type static_size = std::dynamic_extent;

    auto size() const -> std::size_t { return m_data.size(); }
    value_type coeff(index_type i) const { return m_data[i]; }
    value_type& coeff_ref(index_type i) { return m_data[i]; }
    const value_type& const_coeff_ref(index_type i) const { return m_data[i]; }
    value_type* data() { return m_data.data(); }
    const value_type* data() const { return m_data.data(); }

    const auto& container() const { return m_data; }
    auto& container() { return m_data; }

    void resize(index_type s) { m_data.resize(s); }

    using iterator_type = storage_type::iterator;
    using const_iterator_type = storage_type::const_iterator;
    auto begin() -> iterator_type { return m_data.begin(); }
    auto end() -> iterator_type { return m_data.end(); }
    auto begin() const -> const_iterator_type { return m_data.begin(); }
    auto end() const -> const_iterator_type { return m_data.end(); }
    auto cbegin() const -> const_iterator_type { return m_data.begin(); }
    auto cend() const -> const_iterator_type { return m_data.end(); }

    std::span<value_type, static_size> as_std_span() {
        return {data(), size()};
    }
    std::span<const value_type, static_size> as_std_span() const {
        return {data(), size()};
    }

   private:
    storage_type m_data;
};
}  // namespace zipper::storage

#endif
