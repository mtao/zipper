#if !defined(UVL_STORAGE_BASE_HPP)
#define UVL_STORAGE_BASE_HPP
#include <experimental/mdspan>

#include "detail/StorageExtentsTraits.hpp"

namespace uvl::storage {
template <typename Derived_, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
struct StorageBase {
    using Derived = Derived_;
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }

    using StorageExtentsTraits = detail::StorageExtentsTraits<Extents>;
    using mdspan_type =
        uvl::mdspan<Derived, Extents, LayoutPolicy, AccessorPolicy>;
    using extents_type = Extents;
    using mapping_type = typename LayoutPolicy::template mapping<extents_type>;

    const mapping_type& mapping() const { return derived().mapping(); }
    const extents_type& extents() const { return derived().extents(); }
    mdspan_type as_mdspan() { return derived().as_mdspan(); }
    const mdspan_type as_mdspan() const { return derived().as_mdspan(); }

    auto as_span() { return derived().as_span(); }
    const auto as_span() const { return derived().as_span(); }

    template <typename... Indices>
    auto get_index(Indices&&... indices) const -> index_type {
        return mapping()(std::forward<Indices>(indices)...);
    }
};
}  // namespace uvl::storage
#endif
