
#if !defined(UVL_VIEWS_UNARY_IDEMPOTENTVIEW_HPP)
#define UVL_VIEWS_UNARY_IDEMPOTENTVIEW_HPP

#include "UnaryViewBase.hpp"

namespace uvl::views {
namespace unary {
template <concepts::ViewDerived B>
class IdempotentView;


}  // namespace unary

template <concepts::ViewDerived Child>
struct detail::ViewTraits<unary::IdempotentView<Child>>: public uvl::views::unary::detail::DefaultUnaryViewTraits<Child> {};

namespace unary {
template <concepts::ViewDerived B>
class IdempotentView : public UnaryViewBase<IdempotentView< B>, B> {
   public:
    using self_type = IdempotentView< B>;
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using value_type = traits::value_type;
    using child_value_type = traits::base_value_type;

    using Base = UnaryViewBase<self_type, B>;
    using Base::Base;

    decltype(auto) get_value(const child_value_type&value) const
    {
        return value;

    }

};  // namespace unarytemplate<typenameA,typenameB>class AdditionView

}  // namespace unary
}  // namespace uvl::views
#endif
