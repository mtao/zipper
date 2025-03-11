
#if !defined(UVL_DATA_DYNAMICDENSEVIEW_HPP)
#define UVL_DATA_DYNAMICDENSEVIEW_HPP

#include "ViewBase.hpp"
#include "detail/ViewTraits.hpp"
#include "uvl/detail/ExtentsTraits.hpp"

namespace uvl::storage {

template <typename ValueAccessor, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
class DynamicView;

template <typename ValueAccessor, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
struct detail::ViewTraits<
    DynamicView<ValueAccessor, Extents, LayoutPolicy, AccessorPolicy>> {
    using value_accessor_type = ValueAccessor;
    using value_type = typename ValueAccessor::value_type;
    using extents_type = Extents;
    using layout_policy = LayoutPolicy;
    using accessor_policy = AccessorPolicy;
};

template <typename ValueAccessor, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
class DynamicView
    : public ViewBase<
          DynamicView<ValueAccessor, Extents, LayoutPolicy, AccessorPolicy>> {
   public:
    using self_type =
        DynamicView<ValueAccessor, Extents, LayoutPolicy, AccessorPolicy>;
    using Base = ViewBase<self_type>;
    using traits = detail::ViewTraits<self_type>;

    using value_type = traits::value_type;
    using extents_type = traits::extents_type;
    using value_accessor_type = traits::value_accessor_type;
    using extents_traits = uvl::detail::ExtentsTraits<extents_type>;
    using mdspan_type =
        uvl::mdspan<value_type, Extents, LayoutPolicy, AccessorPolicy>;
    using mapping_type = typename LayoutPolicy::template mapping<extents_type>;
    using span_type = extents_traits::template span_type<value_type>;

    const extents_type& extents() const { return m_mapping.extents(); }
    const mapping_type& mapping() const { return m_mapping; }
    mdspan_type as_mdspan() { return mdspan_type{data(), extents()}; }
    const mdspan_type as_mdspan() const {
        return mdspan_type{data(), extents()};
    }
    auto as_span() -> span_type {
        return span_type(data(), value_accessor().size());
    }
    auto as_span() const -> const span_type {
        return span_type(data(), value_accessor().size());
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

    static index_type size_from_extents(const Extents& extents) {
        index_type s = 1;
        for (rank_type j = 0; j < extents.rank(); ++j) {
            s *= extents.extent(j);
        }
        return s;
    }
    index_type size_from_extents() const {
        return size_from_extents(extents());
    }

    template <typename... Args>
    DynamicView(const extents_type& extents, Args&&... args)
        : m_mapping(extents), m_accessor(std::forward<Args>(args)...) {}

   protected:
    value_accessor_type& value_accessor() { return m_accessor; }
    const value_accessor_type& value_accessor() const { return m_accessor; }

   private:
    mapping_type m_mapping;
    value_accessor_type m_accessor;
};

}  // namespace uvl::storage
#endif
