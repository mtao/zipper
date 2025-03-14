#if !defined(UVL_VIEWS_UNARY_SCALARPOWERVIEW_HPP)
#define UVL_VIEWS_UNARY_SCALARPOWERVIEW_HPP

#include <cmath>

#include "detail/CoeffWiseTraits.hpp"
#include "uvl/concepts/ViewDerived.hpp"
#include "uvl/views/ViewBase.hpp"

namespace uvl::views {
namespace unary {
template <concepts::ViewDerived B, typename A>
class ScalarPowerView;

}
template <concepts::ViewDerived B, typename A>
struct detail::ViewTraits<unary::ScalarPowerView<B, A>>
//: public unary::detail::CoeffWiseTraits<A, B> {
//: public detail::ViewTraits<A> {
{
    using Base = detail::ViewTraits<B>;
    using extents_type = typename Base::extents_type;
    using value_type = typename Base::value_type;
    using mapping_type = typename Base::mapping_type;
    constexpr static bool is_writable = false;
};

namespace unary {
template <concepts::ViewDerived B, typename A>
class ScalarPowerView : public ViewBase<ScalarPowerView<B, A>> {
   public:
    using self_type = ScalarPowerView<B, A>;
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using mapping_type = traits::mapping_type;
    using value_type = traits::value_type;

    ScalarPowerView(const B& b, const A& a) : m_view(b), m_pow(a) {}
    using Base = ViewBase<self_type>;
    using Base::extent;

    constexpr const extents_type& extents() const { return m_view.extents(); }

    template <typename... Args>
    value_type coeff(Args&&... idxs) const {
        const auto& view = m_view(std::forward<Args>(idxs)...);
        const auto ret = std::pow(view, m_pow);
        return ret;
    }

   private:
    const B& m_view;
    const A m_pow;
};  // namespace unarytemplate<typenameA,typenameB>class AdditionView

template <concepts::ViewDerived B, typename A>
ScalarPowerView(const B& b, const A& a) -> ScalarPowerView<B, A>;
}  // namespace unary
}  // namespace uvl::views
#endif
