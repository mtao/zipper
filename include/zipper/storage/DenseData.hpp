#if !defined(ZIPPER_STORAGE_DENSEDATA_HPP)
#define ZIPPER_STORAGE_DENSEDATA_HPP
#include "DynamicDenseData.hpp"
#include "LinearAccessorTraits.hpp"
#include "StaticDenseData.hpp"
#include "zipper/detail//ExtentsTraits.hpp"

#include <array>
#include <zipper/types.hpp>

namespace zipper::storage {
template <typename ElementType, index_type N> class DenseData;
template <typename ElementType, index_type N>
  requires(N > 0)
class DenseData<ElementType, N> {

public:
  using element_type = ElementType;
  using value_type = std::remove_cv_t<ElementType>;
  using storage_type = std::array<element_type, N>;
  constexpr static index_type static_size = N;

  constexpr static auto size() -> std::size_t { return N; }
  element_type coeff(index_type i) const { return m_data[i]; }
  element_type &coeff_ref(index_type i) { return m_data[i]; }
  const element_type &const_coeff_ref(index_type i) const { return m_data[i]; }
  element_type *data() { return m_data.data(); }
  const element_type *data() const { return m_data.data(); }

  const auto &container() const { return m_data; }
  auto &container() { return m_data; }

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
template <typename ElementType>
class DenseData<ElementType, std::dynamic_extent> {
public:
  DenseData() = default;
  DenseData(index_type size) : m_data(size) {}
  using element_type = ElementType;
  using value_type = std::remove_cv_t<ElementType>;
  using storage_type = std::vector<element_type>;
  constexpr static index_type static_size = std::dynamic_extent;

  auto size() const -> std::size_t { return m_data.size(); }
  element_type coeff(index_type i) const { return m_data[i]; }
  element_type &coeff_ref(index_type i) { return m_data[i]; }
  const element_type &const_coeff_ref(index_type i) const { return m_data[i]; }
  element_type *data() { return m_data.data(); }
  const element_type *data() const { return m_data.data(); }

  const auto &container() const { return m_data; }
  auto &container() { return m_data; }

  void resize(index_type s) { m_data.resize(s); }

  using iterator_type = storage_type::iterator;
  using const_iterator_type = storage_type::const_iterator;
  auto begin() -> iterator_type { return m_data.begin(); }
  auto end() -> iterator_type { return m_data.end(); }
  auto begin() const -> const_iterator_type { return m_data.begin(); }
  auto end() const -> const_iterator_type { return m_data.end(); }
  auto cbegin() const -> const_iterator_type { return m_data.begin(); }
  auto cend() const -> const_iterator_type { return m_data.end(); }

  std::span<element_type, static_size> as_std_span() {
    return {data(), size()};
  }
  std::span<const element_type, static_size> as_std_span() const {
    return {data(), size()};
  }

private:
  storage_type m_data;
};
template <typename ElementType, std::size_t N>
struct LinearAccessorTraits<DenseData<ElementType, N>>
    : public BasicLinearAccessorTraits<
          AccessFeatures{.is_const = std::is_const_v<ElementType>,
                         .is_reference = true,
                         .is_alias_free = true},
          ShapeFeatures{.is_resizable = N == zipper::dynamic_extent}> {};
} // namespace zipper::storage

#endif
