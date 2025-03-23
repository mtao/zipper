#include <ranges>
#if !defined(UVL_VIEWS_PLAINOBJECTVIEWBASE_HPP)
#define UVL_VIEWS_PLAINOBJECTVIEWBASE_HPP

#include "MappedViewBase.hpp"
#include "ViewBase.hpp"
#include "detail/PlainObjectViewTraits.hpp"
#include "uvl/concepts/TupleLike.hpp"
#include "uvl/concepts/ViewDerived.hpp"
#include "uvl/detail/extents/all_extents_indices.hpp"
namespace uvl::storage {
template <typename ValueType, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
class PlainObjectStorage;
}
namespace uvl::views {
template <typename Derived_>
class PlainObjectViewBase : public MappedViewBase<Derived_> {
   public:
    constexpr static bool IsStatic = uvl::detail::ExtentsTraits<
        typename detail::ViewTraits<Derived_>::extents_type>::is_static;
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

    using ParentType = MappedViewBase<Derived>;
    using ParentType::extent;
    using ParentType::extents;

    using ParentType::ParentType;

    using ParentType::mapping;

    using layout_policy = traits::layout_policy;
    using accessor_policy = traits::accessor_policy;
    using value_accessor_type = traits::value_accessor_type;
    using mapping_type = typename layout_policy::template mapping<extents_type>;
    using span_type = extents_traits::template span_type<value_type>;
    using mdspan_type =
        uvl::mdspan<value_type, extents_type, layout_policy, accessor_policy>;

   public:
    value_accessor_type& accessor() { return derived().accessor(); }
    const value_accessor_type& accessor() const { return derived().accessor(); }
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

    auto as_span() -> span_type { return span_type(accessor().container()); }
    auto as_span() const -> const span_type {
        return span_type(accessor().container());
    }

    template <typename E2>
    void resize(const E2& extents) {
        return derived().resize(extents);
    }

   private:
    template <concepts::ViewDerived V>
    void assign_direct(const V& view) {
        for (const auto& i : uvl::detail::extents::all_extents_indices(extents())) {
            (*this)(i) = view(i);
        }
    }

   public:
    template <concepts::ViewDerived V>
    void assign(const V& view)
        requires(extents_traits::template is_convertable_from<
                 typename detail::ViewTraits<V>::extents_type>())
    {
        using VTraits = detail::ViewTraits<V>;
        constexpr static bool assigning_from_infinite = VTraits::extents_type::rank() == 0;
        constexpr static bool should_resize = !assigning_from_infinite && extents_traits::is_dynamic;
        if constexpr (VTraits::is_coefficient_consistent) {
            if constexpr (should_resize) {
                this->resize(view.extents());
            }
            assign_direct(view);
        } else {
            storage::PlainObjectStorage<value_type, extents_type, layout_policy,
                                        accessor_policy>
                pos(extents_traits::convert_from(view.extents()));
            pos.assign_direct(view);
            if constexpr (should_resize) {
                this->resize(view.extents());
            }
            assign_direct(pos);
        }
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
}  // namespace uvl::views
#endif
