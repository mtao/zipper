#if !defined(ZIPPER_VIEWS_UNARY_UNARYVIEW_HPP)
#define ZIPPER_VIEWS_UNARY_UNARYVIEW_HPP

#include "zipper/concepts/ViewDerived.hpp"
#include "zipper/views/DimensionedViewBase.hpp"
#include "zipper/views/ViewBase.hpp"

namespace zipper::views::unary {

namespace detail {
template <zipper::concepts::QualifiedViewDerived Child,
          bool _holds_extents = false>
struct DefaultUnaryViewTraits
    : public views::detail::DefaultViewTraits<
          typename views::detail::ViewTraits<std::decay_t<Child>>::value_type,
          typename views::detail::ViewTraits<
              std::decay_t<Child>>::extents_type> {
    // to pass a base type to the UnaryViewBase
    constexpr static bool holds_extents = _holds_extents;
    constexpr static bool is_const = std::is_const_v<Child>;
    template <typename Derived>
    using base_type =
        std::conditional_t<holds_extents, DimensionedViewBase<Derived>,
                           ViewBase<Derived>>;
    using base_traits = views::detail::ViewTraits<std::decay_t<Child>>;
    using base_value_type = base_traits::value_type;
    constexpr static bool is_coefficient_consistent =
        base_traits::is_coefficient_consistent;
    constexpr static bool is_value_based = true;
    constexpr static bool is_writable = base_traits::is_writable && !is_const;
};
}  // namespace detail

template <typename Derived, zipper::concepts::QualifiedViewDerived ChildType>
class UnaryViewBase
    : public views::detail::ViewTraits<Derived>::template base_type<Derived> {
   public:
    using self_type = UnaryViewBase<Derived, ChildType>;
    using traits = zipper::views::detail::ViewTraits<Derived>;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
    constexpr static bool holds_extents = traits::holds_extents;
    constexpr static bool is_static = extents_traits::is_static;

    using Base = typename traits::template base_type<Derived>;
    // using Base =
    //     views::detail::ViewTraits<Derived>::template base_type<Derived>;
    using Base::extent;
    constexpr static bool is_value_based = traits::is_value_based;
    constexpr static bool is_const = traits::is_const;
    constexpr static bool is_writable = traits::is_writable;
    using child_type = std::conditional_t<is_writable, ChildType, const ChildType>;

    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }
    using child_value_type = traits::base_value_type;
    UnaryViewBase() = delete;

    UnaryViewBase(const UnaryViewBase&) = default;
    UnaryViewBase(UnaryViewBase&&) = default;
    UnaryViewBase& operator=(const UnaryViewBase&) = delete;
    UnaryViewBase& operator=(UnaryViewBase&&) = delete;
    UnaryViewBase(child_type& b)
        requires(!holds_extents || is_static)
        : m_view(b) {}
    UnaryViewBase(child_type& b, const extents_type& e)
        : Base(e), m_view(b) {}

    constexpr const extents_type& extents() const {
        if constexpr (holds_extents) {
            return Base::extents();
        } else {
            return m_view.extents();
        }
    }

    auto get_value(const child_value_type& value) const -> decltype(auto)
        requires(is_value_based)
    {
        return derived().get_value(value);
    }

    child_type& view()
    {
        return m_view;
    }
    const child_type& view() const { return m_view; }
    template <typename... Args>
    value_type coeff(Args&&... args) const
        requires(is_value_based)
    {
        return get_value(m_view(std::forward<Args>(args)...));
    }

   private:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"
    child_type& m_view;
#pragma GCC diagnostic pop
};

}  // namespace zipper::views::unary
#endif
