#if !defined(UVL_VIEWS_VIEWBASE_HPP)
#define UVL_VIEWS_VIEWBASE_HPP
#include <experimental/mdspan>

#include "detail/ViewTraits.hpp"
#include "uvl/detail/ExtentsTraits.hpp"

namespace uvl::views {
template <typename T>
class ViewBase;
template <typename T>
concept ViewDerived = std::derived_from<T, ViewBase<T>>;

template <typename Derived_>
class ViewBase {
    public:
    using Derived = Derived_;

    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }

     using traits = detail::ViewTraits<Derived>;
     using extents_type = traits::extents_type;
     using value_type = traits::value_type;
    // using extents_traits = uvl::detail::ExtentsTraits<extents_type>;
     using mapping_type = traits::mapping_type;

    constexpr index_type extent(rank_type i) const {
        return extents().extent(i);
    }
     const mapping_type& mapping() const { return derived().mapping(); }
     const extents_type& extents() const { return derived().extents(); }

   public:
    template <typename... Args>
    const value_type& operator()(Args&&... idxs) const {
        return derived().const_coeff_ref(std::forward<Args>(idxs)...);
    }
    template <typename... Args>
    auto&& operator()(Args&&... idxs) {
        return derived().coeff_ref(std::forward<Args>(idxs)...);
    }
};
}  // namespace uvl::views
#endif
