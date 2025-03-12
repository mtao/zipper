

#if !defined(UVL_VIEWS_DYNAMICMAPPEDVIEWBASE_HPP)
#define UVL_VIEWS_DYNAMICMAPPEDVIEWBASE_HPP

#include "DynamicViewBase.hpp"
#include "detail/ViewTraits.hpp"
#include "uvl/detail/ExtentsTraits.hpp"
#include "uvl/detail/convert_extents.hpp"

namespace uvl::views {

template <typename Derived_>
class DynamicMappedViewBase : public DynamicViewBase<Derived_> {
   public:
    using Derived = Derived_;
    using Base = ViewBase<Derived>;
    using traits = detail::ViewTraits<Derived>;
    using Base::extent;

    using value_type = traits::value_type;
    using extents_type = traits::extents_type;
    using value_accessor_type = traits::value_accessor_type;
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

    template <typename E2>
    void resize_extents(const E2& e)
        requires detail::assignable_extents<extents_type, E2>::value
    {
        m_mapping = uvl::detail::convert_extents<extents_type>(e);
    }

   private:
    mapping_type m_mapping;
};

}  // namespace uvl::views
#endif
