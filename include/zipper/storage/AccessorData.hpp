#if !defined(ZIPPER_STORAGE_ACCESSORDATA_HPP)
#define ZIPPER_STORAGE_ACCESSORDATA_HPP
#include "LinearAccessorTraits.hpp"
#include "zipper/detail/no_unique_address.hpp"

#include <type_traits>
#include <utility>
#include <zipper/types.hpp>

namespace zipper::storage {

/// Linear storage that reads elements through an mdspan accessor policy:
/// the k-th element is `accessor.access(data_handle, k)`.
///
/// Used for views of external mdspans whose accessor is not the default
/// one (e.g. scaled, conjugated, atomic or aligned accessors).  Elements are
/// assignable only when the accessor's reference is a plain
/// `element_type &`; proxy references are read-only.
template <typename ElementType, typename Accessor, index_type N>
class AccessorData {
  public:
    using accessor_type = Accessor;
    using element_type = ElementType;
    using value_type = std::remove_cv_t<ElementType>;
    using data_handle_type = typename Accessor::data_handle_type;
    using reference = typename Accessor::reference;
    constexpr static index_type static_size = N;
    constexpr static bool is_const = std::is_const_v<element_type>;
    /// The accessor yields real references into memory.
    constexpr static bool is_reference =
        std::is_same_v<reference, element_type &>;

    static_assert(std::is_same_v<typename Accessor::element_type, ElementType>,
                  "Accessor::element_type must match ElementType");

    AccessorData(data_handle_type handle,
                 std::size_t size,
                 const accessor_type &accessor = {})
      : m_handle(std::move(handle)), m_size(size), m_accessor(accessor) {}

    /// Converting constructor following mdspan's rules, e.g.
    /// default_accessor<T> -> default_accessor<const T>.
    template <typename OtherElementType, typename OtherAccessor, index_type M>
        requires(
            std::is_constructible_v<accessor_type, const OtherAccessor &>
            && std::is_constructible_v<
                data_handle_type,
                const typename OtherAccessor::data_handle_type &>
            && (N == M || N == std::dynamic_extent))
    AccessorData(const AccessorData<OtherElementType, OtherAccessor, M> &other)
      : m_handle(other.data_handle()), m_size(other.size()),
        m_accessor(other.accessor()) {}

    [[nodiscard]] auto size() const -> std::size_t { return m_size; }
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

  private:
    data_handle_type m_handle;
    std::size_t m_size = 0;
    ZIPPER_NO_UNIQUE_ADDRESS accessor_type m_accessor;
};

template <typename ElementType, typename Accessor, index_type N>
struct LinearAccessorTraits<AccessorData<ElementType, Accessor, N>>
  : public BasicLinearAccessorTraits<
        AccessFeatures{AccessorData<ElementType, Accessor, N>::is_const,
                       AccessorData<ElementType, Accessor, N>::is_reference},
        ShapeFeatures::fixed(),
        /*IsContiguous=*/false> {};

} // namespace zipper::storage
#endif
