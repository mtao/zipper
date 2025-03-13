

#if !defined(UVL_VIEWS_DYNAMICMAPPEDVIEWBASE_HPP)
#define UVL_VIEWS_DYNAMICMAPPEDVIEWBASE_HPP

#include "DynamicViewBase.hpp"
#include "detail/ViewTraits.hpp"
#include "uvl/detail/ExtentsTraits.hpp"

namespace uvl::views {

template <typename Derived_>
class DynamicMappedViewBase : public DynamicViewBase<Derived_> {
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
        m_mapping = {e};
    }

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
    mapping_type m_mapping;
};

}  // namespace uvl::views
#endif
