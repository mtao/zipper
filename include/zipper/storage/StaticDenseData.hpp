#if !defined(ZIPPER_STORAGE_STATICDENSEDATA_HPP)
#define ZIPPER_STORAGE_STATICDENSEDATA_HPP

#include <array>
#include <zipper/types.hpp>
#include <span>

namespace zipper::storage {
template <typename ElementType, std::size_t N_>
class StaticDenseData {
    // TODO: clean up downstream logic to not require this hack
    constexpr static std::size_t N = std::max<std::size_t>(
        N_, 1);  // if we have empty extents we have a scalar

   public:
    using element_type = ElementType;
    using value_type = std::remove_cv_t<ElementType>;
    using storage_type = std::array<element_type, N>;
    StaticDenseData() = default;
    StaticDenseData(storage_type data) : m_data(std::move(data)) {}
    constexpr static index_type static_size = N;

    constexpr static auto size() -> std::size_t { return N; }
    element_type coeff(index_type i) const { return m_data[i]; }
    element_type& coeff_ref(index_type i) { return m_data[i]; }
    const element_type& const_coeff_ref(index_type i) const { return m_data[i]; }
    element_type* data() { return m_data.data(); }
    const element_type* data() const { return m_data.data(); }

    const auto& container() const { return m_data; }
    auto& container() { return m_data; }

    using iterator_type = storage_type::iterator;
    using const_iterator_type = storage_type::const_iterator;
    auto begin() -> iterator_type { return m_data.begin(); }
    auto end() -> iterator_type { return m_data.end(); }
    auto begin() const -> const_iterator_type { return m_data.begin(); }
    auto end() const -> const_iterator_type { return m_data.end(); }
    auto cbegin() const -> const_iterator_type { return m_data.begin(); }
    auto cend() const -> const_iterator_type { return m_data.end(); }

    std::span<element_type, static_size> as_std_span() { return container(); }
    std::span<const element_type, static_size> as_std_span() const {
        return container();
    }

   private:
    storage_type m_data;
};
}  // namespace zipper::storage

#endif
