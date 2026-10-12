#if !defined(ZIPPER_STORAGE_SPANDATA_HPP)
#define ZIPPER_STORAGE_SPANDATA_HPP
#include "LinearAccessorTraits.hpp"
#include "zipper/detail/no_unique_address.hpp"

#include <span>
#include <type_traits>
#include <utility>
#include <zipper/types.hpp>

namespace zipper::storage {

namespace detail {
    /// Element count of a SpanData: free for static N, stored otherwise.
    template <std::size_t N>
    struct SpanDataSize {
        SpanDataSize() = default;
        explicit SpanDataSize(std::size_t) {}
        [[nodiscard]] static constexpr auto value() -> std::size_t { return N; }
    };
    template <>
    struct SpanDataSize<std::dynamic_extent> {
        SpanDataSize() = default;
        explicit SpanDataSize(std::size_t n) : m_value(n) {}
        [[nodiscard]] auto value() const -> std::size_t { return m_value; }
        std::size_t m_value = 0;
    };
} // namespace detail

/// Non-owning linear storage over external memory, read through an mdspan
/// accessor policy: element k is `accessor.access(data_handle, k)`.
///
/// With the default accessor this is a std::span: a raw contiguous buffer
/// exposing data(), as_std_span() and iterators.  Any other accessor (e.g.
/// scaled, conjugated, atomic) is honored on every access; elements are then
/// assignable only when the accessor's reference is a plain
/// `element_type &`, and the raw-buffer API is unavailable.
template <typename ElementType,
          std::size_t N,
          typename Accessor = default_accessor_policy<ElementType>>
class SpanData {
  public:
    using accessor_type = Accessor;
    using element_type = ElementType;
    using value_type = std::remove_cv_t<ElementType>;
    using data_handle_type = typename Accessor::data_handle_type;
    using reference = typename Accessor::reference;
    using std_span_type = std::span<element_type, N>;
    constexpr static index_type static_size = N;
    constexpr static bool is_const = std::is_const_v<element_type>;
    /// access(p, k) is p[k], so the storage is a raw contiguous buffer.
    constexpr static bool is_default_accessor =
        std::is_same_v<Accessor, default_accessor_policy<ElementType>>;
    /// The accessor yields real references into memory.
    constexpr static bool is_reference =
        std::is_same_v<reference, element_type &>;

    static_assert(std::is_same_v<typename Accessor::element_type, ElementType>,
                  "Accessor::element_type must match ElementType");

    SpanData()
        requires(N == std::dynamic_extent || N == 0)
    = default;

    SpanData(data_handle_type handle,
             std::size_t size,
             const accessor_type &accessor = {})
      : m_handle(std::move(handle)), m_size(size), m_accessor(accessor) {}

    SpanData(std_span_type s)
        requires(is_default_accessor)
      : SpanData(s.data(), s.size()) {}

    /// Anything a std::span can be built from (containers, other spans,
    /// pointer + count, ...), with std::span's explicitness.
    template <typename... Args>
        requires(is_default_accessor
                 && std::is_constructible_v<std_span_type, Args &&...>
                 && !(sizeof...(Args) == 1
                      && (std::is_same_v<std::remove_cvref_t<Args>, SpanData>
                          && ...)))
    explicit(!(sizeof...(Args) == 1
               && (std::is_convertible_v<Args &&, std_span_type> && ...)))
        SpanData(Args &&...args)
      : SpanData(std_span_type(std::forward<Args>(args)...)) {}

    /// Converting constructor following mdspan's rules, e.g.
    /// SpanData<T, N> -> SpanData<const T, N>.
    template <typename OtherElementType, std::size_t M, typename OtherAccessor>
        requires(!std::is_same_v<SpanData<OtherElementType, M, OtherAccessor>,
                                 SpanData>
                 && std::is_constructible_v<accessor_type, const OtherAccessor &>
                 && std::is_convertible_v<
                     typename OtherAccessor::data_handle_type,
                     data_handle_type>
                 && (N == M || N == std::dynamic_extent))
    SpanData(const SpanData<OtherElementType, M, OtherAccessor> &other)
      : SpanData(other.data_handle(),
                 other.size(),
                 accessor_type(other.accessor())) {}

    [[nodiscard]] auto size() const -> std::size_t { return m_size.value(); }
    [[nodiscard]] auto data_handle() const -> const data_handle_type & {
        return m_handle;
    }
    [[nodiscard]] auto accessor() const -> const accessor_type & {
        return m_accessor;
    }

    auto coeff(index_type i) const -> value_type {
        return static_cast<value_type>(m_accessor.access(m_handle, i));
    }
    auto coeff_ref(index_type i) const -> element_type &
        requires(is_reference && !is_const)
    {
        return m_accessor.access(m_handle, i);
    }
    auto const_coeff_ref(index_type i) const -> const element_type &
        requires(is_reference)
    {
        return m_accessor.access(m_handle, i);
    }

    // ── Raw-buffer API (default accessor only) ─────────────────────────

    auto data() const -> element_type *
        requires(is_default_accessor)
    {
        return m_handle;
    }
    auto as_std_span() const -> std_span_type
        requires(is_default_accessor)
    {
        return std_span_type(m_handle, size());
    }

    using iterator_type = typename std_span_type::iterator;
    using const_iterator_type = iterator_type;
    auto begin() const -> iterator_type
        requires(is_default_accessor)
    {
        return as_std_span().begin();
    }
    auto end() const -> iterator_type
        requires(is_default_accessor)
    {
        return as_std_span().end();
    }
    auto cbegin() const -> const_iterator_type
        requires(is_default_accessor)
    {
        return begin();
    }
    auto cend() const -> const_iterator_type
        requires(is_default_accessor)
    {
        return end();
    }

  private:
    data_handle_type m_handle{};
    ZIPPER_NO_UNIQUE_ADDRESS detail::SpanDataSize<N> m_size;
    ZIPPER_NO_UNIQUE_ADDRESS accessor_type m_accessor;
};

template <typename ElementType, std::size_t N, typename Accessor>
struct LinearAccessorTraits<SpanData<ElementType, N, Accessor>>
  : public BasicLinearAccessorTraits<
        AccessFeatures{SpanData<ElementType, N, Accessor>::is_const,
                       SpanData<ElementType, N, Accessor>::is_reference},
        ShapeFeatures::fixed(),
        /*IsContiguous=*/SpanData<ElementType, N, Accessor>::is_default_accessor> {};
} // namespace zipper::storage
#endif
