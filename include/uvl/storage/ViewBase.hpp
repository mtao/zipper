#if !defined(UVL_STORAGE_BASE_HPP)
#define UVL_STORAGE_BASE_HPP
#include <experimental/mdspan>

#include "detail/ViewTraits.hpp"
#include "uvl/detail/ExtentsTraits.hpp"

namespace uvl::storage {
template <typename Derived_>
class ViewBase {
    using Derived = Derived_;

    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }

    using traits = detail::ViewTraits<Derived>;
    using view_traits = detail::ViewTraits<Derived>;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;
    using layout_policy = traits::layout_policy;
    using accessor_policy = traits::accessor_policy;
    using value_accessor_type = traits::value_accessor_type;
    using extents_traits = uvl::detail::ExtentsTraits<extents_type>;
    using mapping_type = typename layout_policy::template mapping<extents_type>;
    using span_type = extents_traits::template span_type<value_type>;
    using mdspan_type =
        uvl::mdspan<value_type, extents_type, layout_policy, accessor_policy>;

    const mapping_type& mapping() const { return derived().mapping(); }
    const extents_type& extents() const { return derived().extents(); }
    mdspan_type as_mdspan() { return derived().as_mdspan(); }
    const mdspan_type as_mdspan() const { return derived().as_mdspan(); }

   protected:
    template <typename... Indices>
    auto get_index(Indices&&... indices) const -> index_type {
        return mapping()(std::forward<Indices>(indices)...);
    }

   public:
    template <typename... Args>
    value_type operator()(Args&&... idxs) const {
        return derived().coeff(get_index(std::forward<Args>(idxs)...));
    }
    template <typename... Args>
    value_type& operator()(Args&&... idxs) {
        return derived().coeff_ref(get_index(std::forward<Args>(idxs)...));
    }
};
}  // namespace uvl::storage
#endif
