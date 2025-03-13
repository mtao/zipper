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
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }
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

   protected:
    template <std::size_t... Idxs>
    index_type _get_index(concepts::TupleLike auto const& t,
                          std::integer_sequence<index_type, Idxs...>) const {
        return get_index(std::get<Idxs>(t)...);
    }

    template <typename... Indices>
    auto get_index(Indices&&... indices) const -> index_type {
        if constexpr (sizeof...(Indices) == 1 &&
                      (concepts::TupleLike<std::decay_t<Indices>> && ...)) {
            return _get_index(
                indices..., std::make_integer_sequence<std::size_t,
                                                       extents_type::rank()>{});
        } else if constexpr ((std::is_integral_v<std::decay_t<Indices>> &&
                              ...)) {
            static_assert((!concepts::TupleLike<Indices> && ...));
            index_type r = mapping()(std::forward<Indices>(indices)...);
            return r;
        }
    }

   public:
    template <typename... Args>
    auto operator()(Args&&... idxs) const ->

        std::conditional_t<traits::is_writable, const value_type&, value_type> {
        index_type idx = get_index(std::forward<Args>(idxs)...);
        if constexpr (traits::is_writable) {
            return derived().const_coeff_ref(idx);
        } else {
            return derived().coeff(idx);
        }
    }
    template <typename... Args>
    value_type& operator()(Args&&... idxs)
        requires(traits::is_writable)
    {
        index_type idx = get_index(std::forward<Args>(idxs)...);
        return derived().coeff_ref(idx);
    }

   private:
    constexpr static mapping_type s_mapping = {};
};

}  // namespace uvl::views
#endif
