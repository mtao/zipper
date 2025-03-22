#if !defined(UVL_VIEWS_TRACE_HPP)
#define UVL_VIEWS_TRACE_HPP

#include "uvl/concepts/ViewDerived.hpp"
#include "uvl/views/detail/ViewTraits.hpp"
#include "uvl/views/reductions/CoefficientSum.hpp"
#include "uvl/views/unary/DiagonalView.hpp"

namespace uvl::views {
namespace reductions {

template <concepts::ViewDerived View>
class Trace {
   public:
    using self_type = Trace<View>;
    using view_type = View;
    using view_traits = uvl::views::detail::ViewTraits<view_type>;
    using value_type = typename View::value_type;

    Trace(View&& v) : m_view(v) {}
    Trace(const View& v) : m_view(v) {}

    Trace(Trace&& v) = default;
    Trace(const Trace& v) = default;

    value_type operator()() const {
        return reductions::CoefficientSum(unary::DiagonalView(m_view))();
    }

   private:
    const View& m_view;
};

template <concepts::ViewDerived View>
Trace(const View&) -> Trace<View>;

}  // namespace reductions
}  // namespace uvl::views
#endif
