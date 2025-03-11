#if !defined(UVL_DATA_STATICDENSEVIEW_HPP)
#define UVL_DATA_STATICDENSEVIEW_HPP

#include "ViewBase.hpp"
#include "detail/ViewTraits.hpp"
#include "uvl/detail/ExtentsTraits.hpp"

namespace uvl::storage {

template <typename ValueAccessor, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
class StaticView;

template <typename ValueAccessor, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
struct detail::ViewTraits<
    StaticView<ValueAccessor, Extents, LayoutPolicy, AccessorPolicy>> {
    using value_accessor_type = ValueAccessor;
    using value_type = typename ValueAccessor::value_type;
    using extents_type = Extents;
    using layout_policy = LayoutPolicy;
    using accessor_policy = AccessorPolicy;
};

template <typename ValueAccessor, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
class StaticView
    : public ViewBase<
          StaticView<ValueAccessor, Extents, LayoutPolicy, AccessorPolicy>> {
   public:
    using self_type =
        StaticView<ValueAccessor, Extents, LayoutPolicy, AccessorPolicy>;
    using Base = ViewBase<self_type>;
    using traits = detail::ViewTraits<self_type>;

    using value_type = traits::value_type;
    using extents_type = traits::extents_type;
    using layout_policy = traits::layout_policy;
    using accessor_policy = traits::accessor_policy;
    using value_accessor_type = traits::value_accessor_type;
    using extents_traits = uvl::detail::ExtentsTraits<extents_type>;
    using mdspan_type =
        uvl::mdspan<value_type, extents_type, layout_policy, accessor_policy>;
    using mapping_type = typename LayoutPolicy::template mapping<extents_type>;
    using span_type = extents_traits::template span_type<value_type>;

    const extents_type& extents() const { return s_mapping.extents(); }
    const mapping_type& mapping() const { return s_mapping; }
    mdspan_type as_mdspan() { return mdspan_type{data()}; }
    const mdspan_type as_mdspan() const { return mdspan_type{data()}; }

    // TODO: striding needs to be taken care of
    auto as_span() -> span_type { return span_type(data(), data() + size()); }
    auto as_span() const -> const span_type {
        return span_type(data(), data() + size());
    }

    constexpr index_type extent(rank_type i) const {
        return Extents::static_extent(i);
    }

    value_type* data() { return m_accessor.data(); }
    const value_type* data() const { return m_accessor.data(); }
    value_type coeff(index_type i) const { return m_accessor.coeff(i); }
    value_type& coeff_ref(index_type i) { return m_accessor.coeff_ref(i); }
    const value_type& const_coeff_ref(index_type i) const {
        return m_accessor.const_coeff_ref(i);
    }

    constexpr static size_t size() { return extents_traits::static_size; }
    template <typename... Args>
    StaticView(Args&&... args) : m_accessor(std::forward<Args>(args)...) {}

   private:
    constexpr static mapping_type s_mapping = {};
    value_accessor_type m_accessor;
};

}  // namespace uvl::storage
#endif
