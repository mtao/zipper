#if !defined(ZIPPER_VIEWS_DIMENSIONEDVIEWBASE_HPP)
#define ZIPPER_VIEWS_DIMENSIONEDVIEWBASE_HPP

#include "ViewBase.hpp"
namespace zipper::views {
template <typename Derived_>
class DimensionedViewBase : public ViewBase<Derived_> {
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

    const extents_type& extents() const { return m_extents; }

    constexpr index_type extent(rank_type i) const {
        return m_extents.extent(i);
        return extents_type::static_extent(i);
    }

    constexpr size_t size() { return m_extents.size(); }

   private:
    extents_type m_extents;
};
}  // namespace zipper::views
#endif
