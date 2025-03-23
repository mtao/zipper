

#if !defined(UVL_VIEWS_DYNAMICMAPPEDVIEWBASE_HPP)
#define UVL_VIEWS_DYNAMICMAPPEDVIEWBASE_HPP

#include "DynamicViewBase.hpp"
#include "detail/ViewTraits.hpp"
#include "uvl/detail//ExtentsTraits.hpp"

namespace uvl::views {

template <typename Derived_>
class DynamicMappedViewBase : public DynamicViewBase<Derived_> {
   public:
    using Derived = Derived_;
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }
    using Base = ViewBase<Derived>;
    using traits = detail::ViewTraits<Derived>;
    using Base::extent;

    using value_type = traits::value_type;
    using extents_type = traits::extents_type;
    using extents_traits = uvl::detail::ExtentsTraits<extents_type>;
    using mapping_type = traits::mapping_type;

    const extents_type& extents() const { return m_mapping.extents(); }
    const mapping_type& mapping() const { return m_mapping; }

    static index_type size_from_extents(const extents_type& extents) {
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
    DynamicMappedViewBase(const extents_type& extents) : m_mapping(extents) {}

    template <concepts::ExtentsType E2>
    void resize_extents(const E2& e)
        requires(extents_traits::template is_convertable_from<
                 E2>())
    {
        m_mapping = {e};
    }

   protected:
    template <typename... Indices>
    auto get_index(Indices&&... indices) const -> index_type {
        static_assert((std::is_integral_v<std::decay_t<Indices>> && ...));
        static_assert((!concepts::TupleLike<Indices> && ...));
        index_type r = mapping()(std::forward<Indices>(indices)...);
        return r;
    }

   public:
    template <typename... Indices>
    auto coeff(Indices&&... indices) const -> value_type {
        index_type idx = get_index(std::forward<Indices>(indices)...);
        return derived().coeff_linear(idx);
    }
   template <typename... Indices>
   auto coeff_ref(Indices&&... indices)
       -> value_type& requires(traits::is_writable) {
           index_type idx = get_index(std::forward<Indices>(indices)...);
           return derived().coeff_ref_linear(idx);
       }

   template <typename... Indices>
   auto const_coeff_ref(Indices&&... indices) const
       -> const value_type& requires(traits::is_writable) {
           index_type idx = get_index(std::forward<Indices>(indices)...);
           return derived().const_coeff_ref_linear(idx);
       }

   private : mapping_type m_mapping;
};

}  // namespace uvl::views
#endif
