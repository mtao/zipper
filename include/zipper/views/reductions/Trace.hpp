#if !defined(ZIPPER_VIEWS_TRACE_HPP)
#define ZIPPER_VIEWS_TRACE_HPP

#include "zipper/concepts/ViewDerived.hpp"
#include "zipper/views/detail/ViewTraits.hpp"
#include "zipper/views/reductions/CoefficientSum.hpp"
#include "zipper/views/unary/DiagonalView.hpp"

namespace zipper::views {
namespace reductions {

template <concepts::ViewDerived View>
class Trace {
   public:
    using self_type = Trace<View>;
    using view_type = View;
    using view_traits = zipper::views::detail::ViewTraits<view_type>;
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
}  // namespace zipper::views
#endif
