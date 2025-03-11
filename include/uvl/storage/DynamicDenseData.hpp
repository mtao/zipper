#if !defined(UVL_DATA_DYNAMICDENSEDATA_HPP)
#define UVL_DATA_DYNAMICDENSEDATA_HPP

#include <array>
#include <vector>

#include "StorageBase.hpp"
#include "uvl/detail/StorageExtentsTraits.hpp"

namespace uvl::storage {


template <typename value_type_, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
struct DynamicDenseStorage
    : public StorageBase<
          StaticDenseStorage<value_type_, Extents, LayoutPolicy,
                             AccessorPolicy>,
          Extents, LayoutPolicy,
          AccessorPolicy>  // requires(StorageExtentsTraits<Extents>::is_static)
{
    using Base = StorageBase<
        StaticDenseStorage<value_type_, Extents, LayoutPolicy, AccessorPolicy>,
        Extents, LayoutPolicy, AccessorPolicy>;
    using value_type = value_type_;
    using extents_traits = uvl::detail::ExtentsTraits<Extents>;
    using mdspan_type =
        uvl::mdspan<value_type, Extents, LayoutPolicy, AccessorPolicy>;
    using extents_type = Extents;
    using mapping_type = typename LayoutPolicy::template mapping<extents_type>;
    using span_type = StorageExtentsTraits::template span_type<value_type>;

    DynamicDenseStorage(const extents_type& extents)
        : m_mapping(extents), m_data(size()) {}

    DynamicDenseStorage(const mdspan_type& md) : m_mapping(md.extents()) {
        as_mdspan() = md;
    }

    const extents_type& extents() const { return m_mapping.extents(); }
    const mapping_type& mapping() const { return m_mapping; }
    mdspan_type as_mdspan() { return mdspan_type{m_data.data(), extents()}; }
    const mdspan_type as_mdspan() const {
        return mdspan_type{m_data.data(), extents()};
    }
    constexpr index_type extent(rank_type i) const {
        return extents().extent(i);
    }

    auto as_span() -> span_type { return span_type(m_data); }
    auto as_span() const -> const span_type { return span_type(m_data); }

    index_type size() const {
        index_type s = 1;
        for (rank_type j = 0; j < extents().rank(); ++j) {
            s *= extent(j);
        }
        return s;
    }

   private:
    mapping_type m_mapping;
    std::vector<value_type> m_data;
};
template <typename T, typename Extents,
          typename LayoutPolicy = default_layout_policy,
          typename AccessorPolicy = default_accessor_policy<T>>
using DenseStorage = std::conditional_t<
    detail::StorageExtentsTraits<Extents>::is_static,
    StaticDenseStorage<T, Extents, LayoutPolicy, AccessorPolicy>,
    DynamicDenseStorage<T, Extents, LayoutPolicy, AccessorPolicy>>;

}  // namespace uvl::storage
#endif
