#if !defined(UVL_VIEWS_UNARY_SCALARPOWERVIEW_HPP)
#define UVL_VIEWS_UNARY_SCALARPOWERVIEW_HPP

#include <cmath>

#include "UnaryViewBase.hpp"

namespace uvl::views {
namespace unary {
template <concepts::ViewDerived B, typename A>
class ScalarPowerView;

}
template <concepts::ViewDerived Child, typename A>
struct detail::ViewTraits<unary::ScalarPowerView<Child, A>>
    : public uvl::views::unary::detail::DefaultUnaryViewTraits<Child> {
    using value_type = A;
};

namespace unary {
template <concepts::ViewDerived B, typename A>
class ScalarPowerView : public UnaryViewBase<ScalarPowerView<B, A>, B> {
   public:
    using self_type = ScalarPowerView<A, B>;
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;

    using Base = UnaryViewBase<self_type, B>;
    using Base::extent;
    using Base::view;

    ScalarPowerView(const ScalarPowerView&) = default;
    ScalarPowerView(ScalarPowerView&&) = default;
    ScalarPowerView& operator=(const ScalarPowerView&) = default;
    ScalarPowerView& operator=(ScalarPowerView&&) = default;

    ScalarPowerView(const B& b, const A& a) : Base(b), m_exp(a) {}

    const A& exp() const { return m_exp; }
    A& exp() { return m_exp; }

    using child_value_type = traits::base_value_type;
    value_type get_value(const child_value_type& value) const {
        const auto ret = std::pow(value, m_exp);
        return ret;
    }

   private:
    A m_exp;
};

template <typename A, concepts::ViewDerived B>
ScalarPowerView(const A& a, const B& b) -> ScalarPowerView<A, B>;
}  // namespace unary
}  // namespace uvl::views

#endif
