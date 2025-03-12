#if !defined(UVL_VIEWS_STATICVIEWBASE_HPP)
#define UVL_VIEWS_STATICVIEWBASE_HPP

#include "ViewBase.hpp"
#include "detail/ViewTraits.hpp"
#include "uvl/detail/ExtentsTraits.hpp"

namespace uvl::views {



template <typename Derived_>
class StaticViewBase
    : public ViewBase<Derived_> {
   public:
    using Derived =Derived_;
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }
    using Base = ViewBase<Derived>;
    using traits = detail::ViewTraits<Derived>;

    using value_type = traits::value_type;
    using extents_type = traits::extents_type;
    using extents_traits = uvl::detail::ExtentsTraits<extents_type>;

    const extents_type& extents() const { return s_extents; }


    constexpr index_type extent(rank_type i) const {
        return extents_type::static_extent(i);
    }

    constexpr static size_t size() { return extents_traits::static_size; }

   private:
    constexpr static extents_type s_extents = {};
};

}  // namespace uvl::storage
#endif
