#if !defined(UVL_VIEWS_UNARY_UNARYVIEW_HPP)
#define UVL_VIEWS_UNARY_UNARYVIEW_HPP

#include "uvl/concepts/ViewDerived.hpp"
#include "uvl/views/ViewBase.hpp"

namespace uvl::views::unary {


namespace detail {
template <concepts::ViewDerived Child, template <typename> typename Base = ViewBase>
struct DefaultUnaryViewTraits {
    using BaseTraits = views::detail::ViewTraits<Child>;
    using extents_type = typename BaseTraits::extents_type;
    using value_type = BaseTraits::value_type;
    constexpr static bool is_writable = false;

    // to pass a base type to the UnaryViewBase
    template <typename Derived>
    using base_type = Base<Derived>;

};
}

template <typename Derived, concepts::ViewDerived ChildType>
class UnaryViewBase: public views::detail::ViewTraits<Derived>::template base_type<Derived> {
   public:
    using self_type = UnaryViewBase<Derived, ChildType>;
    using traits = uvl::views::detail::ViewTraits<Derived>;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;

    UnaryViewBase(const UnaryViewBase&) = default;
    UnaryViewBase(UnaryViewBase&&) = default;
    UnaryViewBase& operator=(const UnaryViewBase&) = default;
    UnaryViewBase& operator=(UnaryViewBase&&) = default;
    UnaryViewBase(const ChildType& b) : m_view(b) {}
    using Base = views::detail::ViewTraits<Derived>::template base_type<Derived>;
    using Base::extent;

    constexpr const extents_type& extents() const { return m_view.extents(); }

    ChildType& view() { return m_view; }
    const ChildType& view() const { return m_view; }

   private:
    const ChildType& m_view;
};  

}  // namespace uvl::views
#endif
