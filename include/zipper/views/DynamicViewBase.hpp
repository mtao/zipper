
#if !defined(ZIPPER_VIEWS_DYNAMICVIEWBASE_HPP)
#define ZIPPER_VIEWS_DYNAMICVIEWBASE_HPP

#include "ViewBase.hpp"
#include "detail/ViewTraits.hpp"
#include "zipper/detail/ExtentsTraits.hpp"

namespace zipper::views {

template <typename Derived_>
class DynamicViewBase : public ViewBase<Derived_> {
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
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    const extents_type& extents() const { return derived().extents(); }

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
};

}  // namespace zipper::views
#endif
