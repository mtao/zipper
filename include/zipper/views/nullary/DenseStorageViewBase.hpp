#if !defined(ZIPPER_VIEWS_NULLARY_DENSESTORAGEVIEWBASE_HPP)
#define ZIPPER_VIEWS_NULLARY_DENSESTORAGEVIEWBASE_HPP
#include <ranges>

#include "zipper/views/detail/ViewTraits.hpp"
#include "zipper/views/MappedViewBase.hpp"
#include "zipper/views/detail/AssignHelper.hpp"
#include "detail/DenseStorageViewTraits.hpp"
#include "zipper/concepts/TupleLike.hpp"
#include "zipper/concepts/ViewDerived.hpp"
#include "zipper/detail/extents/all_extents_indices.hpp"
namespace zipper::views::nullary {
template <typename Derived_>
class DenseStorageViewBase : public MappedViewBase<Derived_> {
   public:
    using Derived = Derived_;
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }
    using BaseType = ViewBase<Derived>;
    using traits = views::detail::ViewTraits<Derived>;

    using extents_type = traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
    using value_type = traits::value_type;
    constexpr static bool IsStatic = extents_traits::is_static;

    using ParentType = MappedViewBase<Derived>;
    using ParentType::extent;
    using ParentType::extents;

    using ParentType::ParentType;

    //using ParentType::mapping;

    using layout_policy = traits::layout_policy;
    using accessor_policy = traits::accessor_policy;
    using value_accessor_type = traits::value_accessor_type;
    using mapping_type = typename layout_policy::template mapping<extents_type>;
    using span_type = extents_traits::template span_type<value_type>;
    using mdspan_type = zipper::mdspan<value_type, extents_type, layout_policy,
                                       accessor_policy>;

   public:
    value_accessor_type& accessor() { return derived().linear_access(); }
    const value_accessor_type& accessor() const {
        return derived().linear_access();
    }
    value_type* data() { return accessor().data(); }
    const value_type* data() const { return accessor().data(); }
    value_type coeff_linear(index_type i) const { return accessor().coeff(i); }
    value_type& coeff_ref_linear(index_type i) {
        return accessor().coeff_ref(i);
    }
    const value_type& const_coeff_ref_linear(index_type i) const {
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

    auto as_std_span() -> span_type {
        return span_type(accessor().container());
    }
    auto as_std_span() const -> const span_type {
        return span_type(accessor().container());
    }

    template <typename E2>
    void resize(const E2& extents) {
        return derived().resize(extents);
    }

   public:
    template <concepts::ViewDerived V>
    void assign(const V& view)
        requires(extents_traits::template is_convertable_from<
                 typename views::detail::ViewTraits<V>::extents_type>())
    {
        detail::AssignHelper<V, Derived>::assign(view, derived());
    }

    /*
    constexpr static size_t size()
        requires(IsStatic)
    {
        return extents_traits::static_size;
    }
    */
    constexpr size_t size() const { return extents_traits::size(extents()); }
};
}  // namespace zipper::views
#endif
