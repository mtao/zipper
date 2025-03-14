#if !defined(UVL_VIEWS_UNARY_ABSOLUTEVIEW_HPP)
#define UVL_VIEWS_UNARY_ABSOLUTEVIEW_HPP

#include "UnaryViewBase.hpp"

namespace uvl::views {
namespace unary {
template <concepts::ViewDerived Child>
class AbsoluteView;

}  // namespace unary

template <concepts::ViewDerived Child>
struct detail::ViewTraits<unary::AbsoluteView<Child>>: public uvl::views::unary::detail::DefaultUnaryViewTraits<Child> {};

namespace unary {
template <concepts::ViewDerived B>
class AbsoluteView : public UnaryViewBase<AbsoluteView<B>, B> {
   public:
    using self_type = AbsoluteView<B>;
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;

    using Base = UnaryViewBase<self_type, B>;
    using Base::Base;
    using Base::extent;
    using Base::view;


    template <typename... Args>
    value_type coeff(Args&&... idxs) const {
        const auto& value = view()(std::forward<Args>(idxs)...);
        return std::abs(value);
    }
};  

template <concepts::ViewDerived View>
AbsoluteView(View&& view) -> AbsoluteView<View>;
template <concepts::ViewDerived View>
AbsoluteView(const View& view) -> AbsoluteView<View>;
}  // namespace unary
}  // namespace uvl::views
#endif
