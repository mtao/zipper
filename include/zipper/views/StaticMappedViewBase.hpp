#if !defined(ZIPPER_VIEWS_STATICMAPPEDVIEWBASE_HPP)
#define ZIPPER_VIEWS_STATICMAPPEDVIEWBASE_HPP

#include "DimensionedViewBase.hpp"
#include "detail/ViewTraits.hpp"
#include "zipper/detail//ExtentsTraits.hpp"

namespace zipper::views {

// even though this is mapped it can depend
template <typename Derived_>
class StaticMappedViewBase : public ViewBase<Derived_> {
   public:
    using Derived = Derived_;
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }
    using Base = ViewBase<Derived>;
    using traits = detail::ViewTraits<Derived>;

    using value_type = traits::value_type;
    using extents_type = traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
    using mapping_type = traits::mapping_type;
    StaticMappedViewBase(const extents_type& = {}) {}

    const extents_type& extents() const { return s_mapping.extents(); }
    const mapping_type& mapping() const { return s_mapping; }

    constexpr index_type extent(rank_type i) const {
        return extents_type::static_extent(i);
    }

    constexpr static size_t size() { return extents_traits::static_size; }

   protected:
    template <typename... Indices>
    auto get_index(Indices&&... indices) const -> index_type {
        static_assert((std::is_integral_v<std::decay_t<Indices>> && ...));
        static_assert((!concepts::TupleLike<Indices> && ...));
        index_type r = mapping()(std::forward<Indices>(indices)...);
        // spdlog::info("Mapping {} -> ", r);
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

   private : constexpr static mapping_type s_mapping = {};
};

}  // namespace zipper::views
#endif
