

#if !defined(UVL_VIEWS_UNARY_ABSOLUTEVIEW_HPP)
#define UVL_VIEWS_UNARY_ABSOLUTEVIEW_HPP

#include "uvl/concepts/ViewDerived.hpp"
#include "uvl/views/ViewBase.hpp"

namespace uvl::views {
namespace unary {
template <concepts::ViewDerived B>
class AbsoluteView;

}  // namespace unary

template <concepts::ViewDerived B>
struct detail::ViewTraits<unary::AbsoluteView<B>>
//: public unary::detail::CoeffWiseTraits< B> {
//: public detail::ViewTraits<A> {
{
    using Base = detail::ViewTraits<B>;
    using extents_type = typename Base::extents_type;
    using value_type = Base::value_type;
    constexpr static bool is_writable = false;
};

namespace unary {
template <concepts::ViewDerived B>
class AbsoluteView : public ViewBase<AbsoluteView<B>> {
   public:
    using self_type = AbsoluteView<B>;
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;

    AbsoluteView(const AbsoluteView&) = default;
    AbsoluteView(AbsoluteView&&) = default;
    AbsoluteView& operator=(const AbsoluteView&) = default;
    AbsoluteView& operator=(AbsoluteView&&) = default;
    AbsoluteView(const B& b) : m_view(b) {}
    using Base = ViewBase<self_type>;
    using Base::extent;

    constexpr const extents_type& extents() const { return m_view.extents(); }

    template <typename... Args>
    value_type coeff(Args&&... idxs) const {
        const auto& value = m_view(std::forward<Args>(idxs)...);
        return std::abs(value);
    }

   private:
    const B& m_view;
};  // namespace unarytemplate<typenameA,typenameB>class AdditionView

template <concepts::ViewDerived View>
AbsoluteView(View&& view) -> AbsoluteView<View>;
template <concepts::ViewDerived View>
AbsoluteView(const View& view) -> AbsoluteView<View>;
}  // namespace unary
}  // namespace uvl::views
#endif
