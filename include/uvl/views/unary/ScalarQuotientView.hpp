#if !defined(UVL_VIEWS_UNARY_SCALARQUOTIENTVIEW_HPP)
#define UVL_VIEWS_UNARY_SCALARQUOTIENTVIEW_HPP

#include <cmath>

#include "UnaryViewBase.hpp"

namespace uvl::views {
namespace unary {
template <concepts::ViewDerived B, typename A>
class ScalarQuotientView;

}
template <concepts::ViewDerived Child, typename A>
struct detail::ViewTraits<unary::ScalarQuotientView<Child, A>>
    : public uvl::views::unary::detail::DefaultUnaryViewTraits<Child> {
    using value_type = A;
};

namespace unary {
template <concepts::ViewDerived B, typename A>
class ScalarQuotientView : public UnaryViewBase<ScalarQuotientView<B, A>, B> {
   public:
    using self_type = ScalarQuotientView<B, A>;
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;

    using Base = UnaryViewBase<self_type, B>;
    using Base::extent;
    using Base::view;

    ScalarQuotientView(const ScalarQuotientView&) = default;
    ScalarQuotientView(ScalarQuotientView&&) = default;
    ScalarQuotientView& operator=(const ScalarQuotientView&) = default;
    ScalarQuotientView& operator=(ScalarQuotientView&&) = default;

    ScalarQuotientView(const B& b, const A& a) : Base(b), m_quotient(a) {}

    const A& quotient() const { return m_quotient; }
    A& quotient() { return m_quotient; }

    using child_value_type = traits::base_value_type;

    value_type get_value(const child_value_type& value) const {
        const auto ret = value / m_quotient;
        return ret;
    }

   private:
    A m_quotient;
};

template <concepts::ViewDerived A, typename B>
ScalarQuotientView(const A& a, const B& b) -> ScalarQuotientView<A, B>;
}  // namespace unary
}  // namespace uvl::views

#endif
