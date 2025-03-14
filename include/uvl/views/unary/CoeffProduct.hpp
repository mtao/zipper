#if !defined(UVL_VIEWS_UNARY_SCALARPRODUCTVIEW_HPP)
#define UVL_VIEWS_UNARY_SCALARPRODUCTVIEW_HPP

#include "UnaryViewBase.hpp"

namespace uvl::views {
namespace unary {
template <typename A, concepts::ViewDerived B>
class ScalarProductView;

}
template <typename A, concepts::ViewDerived Child>
struct detail::ViewTraits<unary::ScalarProductView<A, Child>>
    : public uvl::views::unary::detail::DefaultUnaryViewTraits<Child> {};

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

    ScalarProductView(const A& a, const B& b) : Base(b), m_lhs(a) {}

    const A& lhs() const { return m_lhs; }
    A& lhs() { return m_lhs; }

    const B& rhs() const { return view(); }
    B& rhs() { return view(); }

    using child_value_type = traits::base_value_type;

    value_type get_value(const child_value_type& value) const {
        return lhs() * value;
    }

   private:
    A m_lhs;
};

template <typename A, concepts::ViewDerived B>
ScalarProductView(const A& a, const B& b) -> ScalarProductView<A, B>;
}  // namespace unary
}  // namespace uvl::views
#endif
