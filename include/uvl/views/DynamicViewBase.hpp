
#if !defined(UVL_VIEWS_DYNAMICVIEWBASE_HPP)
#define UVL_VIEWS_DYNAMICVIEWBASE_HPP

#include "ViewBase.hpp"
#include "detail/ViewTraits.hpp"
#include "uvl/detail/ExtentsTraits.hpp"

namespace uvl::views {

template <typename Derived_>
class DynamicViewBase : public ViewBase<Derived_> {
   public:
    using Derived = Derived_;
    using Base = ViewBase<Derived>;
    using traits = detail::ViewTraits<Derived>;

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
    DynamicViewBase(const extents_type& extents) : m_mapping(extents) {}

   protected:
    value_accessor_type& value_accessor() { return m_accessor; }
    const value_accessor_type& value_accessor() const { return m_accessor; }

   private:
    mapping_type m_mapping;
    value_accessor_type m_accessor;
};

}  // namespace uvl::views
#endif
