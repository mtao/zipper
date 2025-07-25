#if !defined(ZIPPER_VIEWS_NULLARY_NULLARYVIEW_HPP)
#define ZIPPER_VIEWS_NULLARY_NULLARYVIEW_HPP

#include "zipper/views/DimensionedViewBase.hpp"

namespace zipper::views::nullary {

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
    using traits = zipper::views::detail::ViewTraits<Derived>;
    using extents_type = traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
    using value_type = traits::value_type;
    constexpr static bool is_static = extents_traits::is_static;
    constexpr static bool is_coefficient_consistent =
        traits::is_coefficient_consistent;
    constexpr static bool is_value_based = traits::is_value_based;
    using Base =
        views::detail::ViewTraits<Derived>::template base_type<Derived>;
    using Base::extent;

    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }

    constexpr NullaryViewBase(const NullaryViewBase&) = default;
    constexpr NullaryViewBase(NullaryViewBase&&) = default;
    constexpr NullaryViewBase()
        requires(is_static)
    = default;

    constexpr NullaryViewBase(const extents_type& e) : Base(e) {}

    constexpr value_type get_value() const
        requires(is_value_based)
    {
        return derived().get_value();
    }

    template <typename... Args>
    constexpr value_type coeff(Args&&...) const
        requires(is_value_based)
    {
        return get_value();
    }
};

}  // namespace zipper::views::nullary
#endif
