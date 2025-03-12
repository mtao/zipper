#if !defined(UVL_VIEWS_VIEWBASE_HPP)
#define UVL_VIEWS_VIEWBASE_HPP
#include <experimental/mdspan>

#include "detail/ViewTraits.hpp"
#include "uvl/detail/ExtentsTraits.hpp"

namespace uvl::views {

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
     using array_type = std::array<index_type, extents_type::rank()>;
    // using extents_traits = uvl::detail::ExtentsTraits<extents_type>;

    constexpr index_type extent(rank_type i) const {
        return extents().extent(i);
    }
     const extents_type& extents() const { return derived().extents(); }

   public:
    template <typename... Args>
    const value_type& operator()(Args&&... idxs) const {
        return derived()(std::forward<Args>(idxs)...);
    }
    template <typename... Args>
    auto&& operator()(Args&&... idxs) {
        return derived()(std::forward<Args>(idxs)...);
    }

    template <std::size_t... Idxs>
    const value_type& _call( const array_type& t) const {
        return derived()(t[Idxs]...);
    }
    template <std::size_t... Idxs>
    auto&& _call(const array_type& t) {
        return derived()(t[Idxs]...);
    }
    const value_type& operator()( const array_type& t) const {
        return _call(t,std::make_integer_sequence<std::size_t,extents_type::rank()>{});
    }
    auto&& operator()(const array_type& t) {
        return _call(t,std::make_integer_sequence<std::size_t,extents_type::rank()>{});
    }
};
}  // namespace uvl::views
#endif
