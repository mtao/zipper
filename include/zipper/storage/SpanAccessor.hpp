#if !defined(ZIPPER_STORAGE_SPANACCESSOR_HPP)
#define ZIPPER_STORAGE_SPANACCESSOR_HPP


#include "zipper/types.hpp"
namespace zipper::storage {
template <typename ValueType, std::size_t N>
class SpanAccessor {
   public:
    using value_type = ValueType;
    using std_span_type = std::span<value_type, N>;

    SpanAccessor(const std_span_type& s): m_data(s) {}


    constexpr auto size() -> std::size_t { return m_data.size(); }
    value_type coeff(index_type i) const { return m_data[i]; }
    value_type& coeff_ref(index_type i) { return m_data[i]; }
    const value_type& const_coeff_ref(index_type i) const { return m_data[i]; }
    value_type* data() { return m_data.data(); }
    const value_type* data() const { return m_data.data(); }

    const auto& container() const { return m_data; }
    auto& container() { return m_data; }

   private:
    std_span_type m_data;
};

}  // namespace zipper::storage
#endif
