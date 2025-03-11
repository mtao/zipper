#if !defined(UVL_VIEWS_PLAINOBJECTVIEWBASE_HPP)
#define UVL_VIEWS_PLAINOBJECTVIEWBASE_HPP

#include "DynamicViewBase.hpp"
#include "StaticViewBase.hpp"
#include "ViewBase.hpp"
#include "detail/PlainObjectViewTraits.hpp"
namespace uvl::views {
template <typename Derived_,
          bool IsStatic = uvl::detail::ExtentsTraits<
              typename detail::ViewTraits<Derived_>::extents_type>::is_static>
class PlainObjectViewBase
    : public

      std::conditional_t<IsStatic, StaticViewBase<Derived_>,
                         DynamicViewBase<Derived_>> {
    static_assert(
        IsStatic ==
        uvl::detail::ExtentsTraits<
            typename detail::ViewTraits<Derived_>::extents_type>::is_static);
    using Derived = Derived_;
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }
    using BaseType = ViewBase<Derived>;
    using traits = detail::ViewTraits<Derived>;

    using extents_type = traits::extents_type;
    using extents_traits = uvl::detail::ExtentsTraits<extents_type>;
    using value_type = traits::value_type;

    using ParentType = std::conditional_t<IsStatic, StaticViewBase<Derived_>,
                                          DynamicViewBase<Derived_>>;
    using ParentType::extents;

    using ParentType::mapping;

    using layout_policy = traits::layout_policy;
    using accessor_policy = traits::accessor_policy;
    using value_accessor_type = traits::value_accessor_type;
    using mapping_type = typename layout_policy::template mapping<extents_type>;
    using span_type = extents_traits::template span_type<value_type>;
    using mdspan_type =
        uvl::mdspan<value_type, extents_type, layout_policy, accessor_policy>;

   protected:
    template <typename... Indices>
    auto get_index(Indices&&... indices) const -> index_type {
        return mapping()(std::forward<Indices>(indices)...);
    }

   public:
    value_accessor_type& accessor() { return derived().accessor(); }
    const value_accessor_type& accessor() const { return derived().accessor(); }
    value_type* data() { return accessor().data(); }
    const value_type* data() const { return accessor().data(); }
    value_type coeff(index_type i) const { return accessor().coeff(i); }
    value_type& coeff_ref(index_type i) { return accessor().coeff_ref(i); }
    const value_type& const_coeff_ref(index_type i) const {
        return accessor().const_coeff_ref(i);
    }

    mdspan_type as_mdspan() {
        if constexpr (IsStatic) {
            return mdspan_type(data());
        } else {
            return mdspan_type(data(), extents());
        }
    }
    const mdspan_type as_mdspan() const {
        if constexpr (IsStatic) {
            return mdspan_type(data());
        } else {
            return mdspan_type(data(), extents());
        }
    }

    auto as_span() -> span_type { return span_type(accessor().container()); }
    auto as_span() const -> const span_type {
        return span_type(accessor().container());
    }

    constexpr index_type extent(rank_type i) const {
        return extents().static_extent(i);
    }

    /*
    constexpr static size_t size()
        requires(IsStatic)
    {
        return extents_traits::static_size;
    }
    */
    constexpr size_t size() const { return extents_traits::size(extents()); }

    template <typename... Args>
    const value_type& operator()(Args&&... idxs) const {
        return derived().coeff(get_index(std::forward<Args>(idxs)...));
    }
    template <typename... Args>
    value_type& operator()(Args&&... idxs) {
        return derived().coeff_ref(get_index(std::forward<Args>(idxs)...));
    }
};
}  // namespace uvl::views
#endif
