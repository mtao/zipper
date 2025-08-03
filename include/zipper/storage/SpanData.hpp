#if !defined(ZIPPER_STORAGE_SPANDATA_HPP)
#define ZIPPER_STORAGE_SPANDATA_HPP

#include <zipper/types.hpp>
namespace zipper::storage {
template <typename ValueType, std::size_t N>
class SpanData {
   public:
    using value_type = ValueType;
    using std_span_type = std::span<value_type, N>;
    constexpr static index_type static_size = N;

    SpanData(const std_span_type& s) : m_data(s) {}

    constexpr auto size() const noexcept -> std::size_t { return m_data.size(); }
    value_type coeff(index_type i) const { return m_data[i]; }
    value_type& coeff_ref(index_type i) { return m_data[i]; }
    const value_type& const_coeff_ref(index_type i) const { return m_data[i]; }
    value_type* data() noexcept { return m_data.data(); }
    const value_type* data() const noexcept { return m_data.data(); }

    const auto& container() const { return m_data; }
    auto& container() { return m_data; }

    auto as_stl_span() { return m_data; }
    auto as_stl_span() const { return m_data; }

    // TODO: for cpp<23 iterator_type and const_iterator_type are different.
    // feeling very lazy as this discrepency disappears with cpp23
#if defined(__cpp_lib_ranges_as_const)
    using iterator_type = std_span_type::iterator;
    using const_iterator_type = std_span_type::const_iterator;
#else
    using iterator_type = std_span_type::iterator;
    using const_iterator_type = iterator_type;
#endif
    auto begin() noexcept -> iterator_type { return m_data.begin(); }
    auto end() noexcept -> iterator_type { return m_data.end(); }
    auto begin() const noexcept -> const_iterator_type { return m_data.begin(); }
    auto end() const noexcept -> const_iterator_type { return m_data.end(); }
    auto cbegin() const noexcept -> const_iterator_type { return m_data.begin(); }
    auto cend() const noexcept -> const_iterator_type { return m_data.end(); }

   private:
    std_span_type m_data;
};

}  // namespace zipper::storage
#endif
