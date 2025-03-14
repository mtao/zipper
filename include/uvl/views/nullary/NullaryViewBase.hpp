#if !defined(UVL_VIEWS_NULLARY_NULLARYVIEW_HPP)
#define UVL_VIEWS_NULLARY_NULLARYVIEW_HPP

#include "uvl/views/DimensionedViewBase.hpp"

namespace uvl::views::nullary {

namespace detail {
template <typename T, index_type... Indices>
struct DefaultNullaryViewTraits
    : public views::detail::DefaultViewTraits<T, extents<Indices...>> {
    constexpr static bool is_coefficient_consistent = true;
    constexpr static bool is_value_based = true;

    // to pass a base type to the NullaryViewBase
    template <typename Derived>
    using base_type = DimensionedViewBase<Derived>;
};
}  // namespace detail

template <typename Derived, typename T, index_type... Indices>
class NullaryViewBase
    : public views::detail::ViewTraits<Derived>::template base_type<Derived> {
   public:
    using self_type = NullaryViewBase<Derived, T, Indices...>;
    using traits = uvl::views::detail::ViewTraits<Derived>;
    using extents_type = traits::extents_type;
    using extents_traits = uvl::detail::ExtentsTraits<extents_type>;
    using value_type = traits::value_type;
    constexpr static bool is_static = extents_traits::is_static;
    constexpr static bool is_coefficient_consistent =
        traits::is_coefficient_consistent;
    constexpr static bool is_value_based = traits::is_value_based;

    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }

    NullaryViewBase(const NullaryViewBase&) = default;
    NullaryViewBase(NullaryViewBase&&) = default;
    NullaryViewBase& operator=(const NullaryViewBase&) = default;
    NullaryViewBase& operator=(NullaryViewBase&&) = default;
    NullaryViewBase()
        requires(is_static)
    {}

    NullaryViewBase(const extents_type& e) : m_extents(e) {}
    using Base =
        views::detail::ViewTraits<Derived>::template base_type<Derived>;
    using Base::extent;

    const extents_type& extents() const { return m_extents; }

    value_type get_value() const
        requires(is_value_based)
    {
        return derived().get_value();
    }

    template <typename... Args>
    value_type coeff(Args&&...) const
        requires(is_value_based)
    {
        return get_value();
    }

   private:
    extents_type m_extents;
};

}  // namespace uvl::views::nullary
#endif
