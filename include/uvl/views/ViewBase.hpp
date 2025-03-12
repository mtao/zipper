#if !defined(UVL_VIEWS_VIEWBASE_HPP)
#define UVL_VIEWS_VIEWBASE_HPP
#include <experimental/mdspan>

#include "detail/ViewTraits.hpp"
#include "uvl/concepts/TupleLike.hpp"
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
    constexpr static bool is_writable = traits::is_writable;
    using array_type = std::array<index_type, extents_type::rank()>;
    // using extents_traits = uvl::detail::ExtentsTraits<extents_type>;

    constexpr index_type extent(rank_type i) const {
        return extents().extent(i);
    }
    const extents_type& extents() const { return derived().extents(); }

   public:
    template <typename... Args>
    const value_type operator()(Args&&... idxs) const

        requires((std::is_convertible_v<Args, index_type> && ...))
    {
        return derived().coeff(std::forward<Args>(idxs)...);
    }
    template <typename... Args>
    value_type& operator()(Args&&... idxs)

        requires((std::is_convertible_v<Args, index_type> && ...) &&
                 is_writable)
    {
        return derived().coeff_ref(std::forward<Args>(idxs)...);
    }

    template <std::size_t... Idxs>
    value_type _call(concepts::TupleLike auto const& t,
                     std::integer_sequence<std::size_t, Idxs...>) const {
        return derived().coeff(std::get<Idxs>(t)...);
    }
    template <std::size_t... Idxs>
    value_type& _call(concepts::TupleLike auto const& t,
                      std::integer_sequence<std::size_t, Idxs...>)
        requires(is_writable)
    {
        return derived().coeff_ref(std::get<Idxs>(t)...);
    }
    value_type operator()(concepts::TupleLike auto const& t) const {
        return _call(
            t, std::make_integer_sequence<std::size_t, extents_type::rank()>{});
    }
    auto& operator()(concepts::TupleLike auto const& t)
        requires(is_writable)
    {
        return _call(
            t, std::make_integer_sequence<std::size_t, extents_type::rank()>{});
    }
};
}  // namespace uvl::views
#endif
