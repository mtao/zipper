#if !defined(UVL_VIEWS_UNARY_NEGATEVIEW_HPP)
#define UVL_VIEWS_UNARY_NEGATEVIEW_HPP

#include "UnaryViewBase.hpp"

namespace uvl::views {
namespace unary {
template <concepts::ViewDerived Child>
class NegateView;

}  // namespace unary

template <concepts::ViewDerived Child>
struct detail::ViewTraits<unary::NegateView<Child>>
    : public uvl::views::unary::detail::DefaultUnaryViewTraits<Child> {};

namespace unary {
template <concepts::ViewDerived B>
class NegateView : public UnaryViewBase<NegateView<B>, B> {
   public:
    using self_type = NegateView<B>;
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;

    using Base = UnaryViewBase<self_type, B>;
    using Base::Base;
    using Base::extent;
    using Base::view;

    using child_value_type = traits::base_value_type;

    value_type get_value(const child_value_type& value) const { return -value; }
};

template <concepts::ViewDerived View>
NegateView(View&& view) -> NegateView<View>;
template <concepts::ViewDerived View>
NegateView(const View& view) -> NegateView<View>;
}  // namespace unary
}  // namespace uvl::views
#endif
