#if !defined(UVL_VIEWS_UNARY_SCALARPRODUCTVIEW_HPP)
#define UVL_VIEWS_UNARY_SCALARPRODUCTVIEW_HPP

#include "detail/CoeffWiseTraits.hpp"
#include "uvl/concepts/ViewDerived.hpp"
#include "uvl/views/ViewBase.hpp"

namespace uvl::views {
namespace unary {
template <typename A, concepts::ViewDerived B>
class ScalarProductView;

}
template <typename A, concepts::ViewDerived B>
struct detail::ViewTraits<unary::ScalarProductView<A, B>>
//: public unary::detail::CoeffWiseTraits<A, B> {
//: public detail::ViewTraits<A> {
{
    using Base = detail::ViewTraits<B>;
    using extents_type = typename Base::extents_type;
    using value_type = typename Base::value_type;
    constexpr static bool is_writable = false;
};

namespace unary {
template <typename A, concepts::ViewDerived B>
class ScalarProductView : public ViewBase<ScalarProductView<A, B>> {
   public:
    using self_type = ScalarProductView<A, B>;
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;

    ScalarProductView(const A& a, const B& b) : m_lhs(a), m_rhs(b) {}
    using Base = ViewBase<self_type>;
    using Base::extent;

    constexpr const extents_type& extents() const { return m_rhs.extents(); }

    template <typename... Args>
    value_type coeff(Args&&... idxs) const {
        const auto& rhs = m_rhs(std::forward<Args>(idxs)...);
        const auto ret = m_lhs * rhs;
        return ret;
    }

   private:
    const A m_lhs;
    const B& m_rhs;
};  // namespace unarytemplate<typenameA,typenameB>class AdditionView

template <typename A, concepts::ViewDerived B>
ScalarProductView(const A& a, const B& b) -> ScalarProductView<A, B>;
}  // namespace unary
}  // namespace uvl::views
#endif
