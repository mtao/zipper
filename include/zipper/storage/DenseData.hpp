#if !defined(ZIPPER_STORAGE_DENSEDATA_HPP)
#define ZIPPER_STORAGE_DENSEDATA_HPP
#include "LinearAccessorTraits.hpp"
#include "zipper/detail/ExtentsTraits.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>
#include <zipper/types.hpp>

namespace zipper::storage {
template <typename ElementType, index_type N>
class DenseData;
template <typename ElementType, index_type N>
    requires(N > 0)
class DenseData<ElementType, N> {
  public:
    using element_type = ElementType;
    using value_type = std::remove_cv_t<ElementType>;
    using storage_type = std::array<element_type, N>;
    constexpr static index_type static_size = N;

    DenseData() = default;
    DenseData(storage_type data) : m_data(std::move(data)) {}
    /// Default-initializes the elements (indeterminate for trivial types).
    /// See zipper::uninitialized_t.
    explicit DenseData(uninitialized_t) {}

    constexpr static auto size() -> std::size_t { return N; }
    element_type coeff(index_type i) const { return data()[i]; }
    element_type &coeff_ref(index_type i) { return data()[i]; }
    const element_type &const_coeff_ref(index_type i) const {
        return data()[i];
    }
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
    using element_type = ElementType;
    using value_type = std::remove_cv_t<ElementType>;
    constexpr static index_type static_size = std::dynamic_extent;
    using iterator_type = element_type *;
    using const_iterator_type = const element_type *;

    DenseData() = default;
    /// Allocates `size` value-initialized (zero-filled) elements.
    explicit DenseData(index_type size)
      : m_data(std::make_unique<value_type[]>(size)), m_size(size),
        m_capacity(size) {}
    /// Allocates `size` default-initialized elements (indeterminate for
    /// trivial types). See zipper::uninitialized_t.
    DenseData(uninitialized_t, index_type size)
      : m_data(allocate<false>(size)), m_size(size), m_capacity(size) {}

    DenseData(const DenseData &o)
      : m_data(allocate<false>(o.m_size)),
        m_size(o.m_size), m_capacity(o.m_size) {
        std::ranges::copy(o, begin());
    }
    DenseData(DenseData &&o) noexcept
      : m_data(std::move(o.m_data)), m_size(std::exchange(o.m_size, 0)),
        m_capacity(std::exchange(o.m_capacity, 0)) {}
    auto operator=(const DenseData &o) -> DenseData & {
        if (this != &o) {
            if (o.m_size > m_capacity) {
                m_data = allocate<false>(o.m_size);
                m_capacity = o.m_size;
            }
            m_size = o.m_size;
            std::ranges::copy(o, begin());
        }
        return *this;
    }
    auto operator=(DenseData &&o) noexcept -> DenseData & {
        m_data = std::move(o.m_data);
        m_size = std::exchange(o.m_size, 0);
        m_capacity = std::exchange(o.m_capacity, 0);
        return *this;
    }
    ~DenseData() = default;

    auto size() const -> std::size_t { return m_size; }
    element_type coeff(index_type i) const { return data()[i]; }
    element_type &coeff_ref(index_type i) { return data()[i]; }
    const element_type &const_coeff_ref(index_type i) const {
        return data()[i];
    }
    element_type *data() { return m_data.get(); }
    const element_type *data() const { return m_data.get(); }

    auto container() const -> const DenseData & { return *this; }
    auto container() -> DenseData & { return *this; }

    /// Resizes, preserving the common prefix; new elements are zero-filled.
    /// Shrinking (or growing within capacity) does not reallocate.
    void resize(index_type s) {
        const std::size_t old = m_size;
        const bool reallocates = s > m_capacity;
        resize_impl<true>(s);
        if (!reallocates) {
            // Slots within capacity hold live objects from a previous
            // (larger) size; replace them with value-initialized ones.
            for (value_type *p = m_data.get() + old; p < m_data.get() + s;
                 ++p) {
                std::destroy_at(p);
                std::construct_at(p);
            }
        }
    }
    /// Resizes, preserving the common prefix; new elements are
    /// default-initialized (indeterminate for trivial types) and, when
    /// growing within capacity, may hold stale values. Shrinking (or growing
    /// within capacity) does not reallocate. See zipper::uninitialized_t.
    void resize(uninitialized_t, index_type s) { resize_impl<false>(s); }

    auto capacity() const -> std::size_t { return m_capacity; }

    auto begin() -> iterator_type { return data(); }
    auto end() -> iterator_type { return data() + m_size; }
    auto begin() const -> const_iterator_type { return data(); }
    auto end() const -> const_iterator_type { return data() + m_size; }
    auto cbegin() const -> const_iterator_type { return data(); }
    auto cend() const -> const_iterator_type { return data() + m_size; }

    std::span<element_type, static_size> as_std_span() {
        return {data(), size()};
    }
    std::span<const element_type, static_size> as_std_span() const {
        return {data(), size()};
    }

  private:
    // Sets the size to `s`, reallocating only if `s` exceeds capacity. On
    // reallocation the current prefix is moved over and the remaining slots
    // are value-initialized (ZeroFill) or default-initialized.
    template <bool ZeroFill>
    void resize_impl(std::size_t s) {
        if (s > m_capacity) {
            auto d = allocate<ZeroFill>(s);
            std::move(data(), data() + m_size, d.get());
            m_data = std::move(d);
            m_capacity = s;
        }
        m_size = s;
    }
    template <bool ZeroFill>
    static auto allocate(std::size_t s) -> std::unique_ptr<value_type[]> {
        if constexpr (ZeroFill) {
            return std::make_unique<value_type[]>(s);
        } else {
            return std::make_unique_for_overwrite<value_type[]>(s);
        }
    }

    std::unique_ptr<value_type[]> m_data;
    std::size_t m_size = 0;
    std::size_t m_capacity = 0;
};
template <typename ElementType, std::size_t N>
struct LinearAccessorTraits<DenseData<ElementType, N>>
  : public BasicLinearAccessorTraits<
        AccessFeatures{std::is_const_v<ElementType>, true},
        ShapeFeatures{N == zipper::dynamic_extent}> {};
} // namespace zipper::storage

#endif
