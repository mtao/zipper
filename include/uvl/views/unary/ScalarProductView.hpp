
#if !defined(UVL_VIEWS_UNARY_SCALARPRODUCTVIEW_HPP)
#define UVL_VIEWS_UNARY_SCALARPRODUCTVIEW_HPP

#include "UnaryViewBase.hpp"
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
    : public unary::detail::DefaultUnaryViewTraits<B> {};

namespace unary {
template <typename A, concepts::ViewDerived B>
class ScalarProductView : public UnaryViewBase<ScalarProductView<A, B>, B> {
   public:
    using self_type = ScalarProductView<A, B>;
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;

    using Base = UnaryViewBase<self_type, B>;
    using Base::extent;
    using Base::view;

    ScalarProductView(const ScalarProductView&) = default;
    ScalarProductView(ScalarProductView&&) = default;
    ScalarProductView& operator=(const ScalarProductView&) = default;
    ScalarProductView& operator=(ScalarProductView&&) = default;
    ScalarProductView(const A& a, const B& b) : Base(b), m_lhs(a) {}

    template <typename... Args>
    value_type coeff(Args&&... idxs) const {
        const auto& rhs = view()(std::forward<Args>(idxs)...);
        const auto ret = m_lhs * rhs;
        return ret;
    }

   private:
    const A m_lhs;
};  // namespace unarytemplate<typenameA,typenameB>class AdditionView

template <typename A, concepts::ViewDerived B>
ScalarProductView(const A& a, const B& b) -> ScalarProductView<A, B>;
}  // namespace unary
}  // namespace uvl::views
#endif
