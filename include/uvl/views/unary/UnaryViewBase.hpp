#if !defined(UVL_VIEWS_UNARY_UNARYVIEW_HPP)
#define UVL_VIEWS_UNARY_UNARYVIEW_HPP

#include "uvl/concepts/ViewDerived.hpp"
#include "uvl/views/ViewBase.hpp"

namespace uvl::views::unary {

namespace detail {
template <concepts::ViewDerived Child,
          template <typename> typename Base = ViewBase>
struct DefaultUnaryViewTraits
    : public views::detail::DefaultViewTraits<
          typename views::detail::ViewTraits<Child>::value_type,
          typename views::detail::ViewTraits<Child>::extents_type> {
    // to pass a base type to the UnaryViewBase
    template <typename Derived>
    using base_type = Base<Derived>;
    using base_traits = views::detail::ViewTraits<Child>;
    using base_value_type = base_traits::value_type;
    constexpr static bool is_coefficient_consistent =
        base_traits::is_coefficient_consistent;
    constexpr static bool holds_extents = false;
    constexpr static bool is_value_based = true;
    constexpr static bool is_const = false;
};
}  // namespace detail

template <typename Derived, concepts::ViewDerived ChildType>
class UnaryViewBase : public views::ViewBase<Derived> {
    //: public views::detail::ViewTraits<Derived>::template base_type<Derived> {
   public:
    using self_type = UnaryViewBase<Derived, ChildType>;
    using traits = uvl::views::detail::ViewTraits<Derived>;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;

    using Base = views::ViewBase<Derived>;
    // using Base =
    //     views::detail::ViewTraits<Derived>::template base_type<Derived>;
    using Base::extent;
    constexpr static bool is_value_based = traits::is_value_based;
    constexpr static bool is_const = traits::is_const;

    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }
    using child_value_type = traits::base_value_type;

    UnaryViewBase(const UnaryViewBase&) = default;
    UnaryViewBase(UnaryViewBase&&) = default;
    UnaryViewBase& operator=(const UnaryViewBase&) = default;
    UnaryViewBase& operator=(UnaryViewBase&&) = default;
    UnaryViewBase(const ChildType& b) : m_view(b) {}

    constexpr const extents_type& extents() const { return m_view.extents(); }

    auto get_value(const child_value_type& value) const -> decltype(auto)
        requires(is_value_based)
    {
        return derived().get_value(value);
    }

    ChildType& view()
        requires(!is_const)
    {
        return const_cast<ChildType&>(m_view);
    }
    const ChildType& view() const { return m_view; }
    template <typename... Args>
    value_type coeff(Args&&... args) const
        requires(is_value_based)
    {
        return get_value(m_view(std::forward<Args>(args)...));
    }

   private:
    const ChildType& m_view;
};

}  // namespace uvl::views::unary
#endif
