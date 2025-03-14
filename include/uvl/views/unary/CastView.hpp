
#if !defined(UVL_VIEWS_UNARY_CASTVIEW_HPP)
#define UVL_VIEWS_UNARY_CASTVIEW_HPP

#include "UnaryViewBase.hpp"

namespace uvl::views {
namespace unary {
template <typename A, concepts::ViewDerived B>
class CastView;

template <typename A, concepts::ViewDerived B>
auto cast(const B& b) {
    return CastView<A, B>(b);
}

}  // namespace unary

template <typename A, concepts::ViewDerived Child>
struct detail::ViewTraits<unary::CastView<A,Child>>: public uvl::views::unary::detail::DefaultUnaryViewTraits<Child> {

    using value_type = A;

};

namespace unary {
template <typename A, concepts::ViewDerived B>
class CastView : public UnaryViewBase<CastView<A, B>, B> {
   public:
    using self_type = CastView<A, B>;
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
        return static_cast<A>(value);
    }
};  // namespace unarytemplate<typenameA,typenameB>class AdditionView

}  // namespace unary
}  // namespace uvl::views
#endif
