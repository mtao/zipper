#if !defined(UVL_VIEWS_STATICMAPPEDVIEWBASE_HPP)
#define UVL_VIEWS_STATICMAPPEDVIEWBASE_HPP

#include "StaticViewBase.hpp"
#include "detail/ViewTraits.hpp"
#include "uvl/detail/ExtentsTraits.hpp"

namespace uvl::views {

template <typename Derived_>
class StaticMappedViewBase : public StaticViewBase<Derived_> {
   public:
    using Derived = Derived_;
    using Base = ViewBase<Derived>;
    using traits = detail::ViewTraits<Derived>;

    using value_type = traits::value_type;
    using extents_type = traits::extents_type;
    using extents_traits = uvl::detail::ExtentsTraits<extents_type>;
    using mapping_type = traits::mapping_type;
    StaticMappedViewBase(const extents_type& = {}) {}

    const extents_type& extents() const { return s_mapping.extents(); }
    const mapping_type& mapping() const { return s_mapping; }

    constexpr index_type extent(rank_type i) const {
        return extents_type::static_extent(i);
    }

    constexpr static size_t size() { return extents_traits::static_size; }

   private:
    constexpr static mapping_type s_mapping = {};
};

}  // namespace uvl::views
#endif
