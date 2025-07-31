#if !defined(ZIPPER_VIEWS_TRACE_HPP)
#define ZIPPER_VIEWS_TRACE_HPP

#include "zipper/concepts/ViewDerived.hpp"
#include "zipper/views/detail/ViewTraits.hpp"
#include "zipper/views/reductions/CoefficientSum.hpp"
#include "zipper/views/unary/DiagonalView.hpp"

namespace zipper::views {
namespace reductions {

template <zipper::concepts::QualifiedViewDerived View>
class Trace {
   public:
    using self_type = Trace<View>;
    using view_type = View;
    using view_traits =
        zipper::views::detail::ViewTraits<std::decay_t<view_type>>;
    using value_type = typename view_traits::value_type;

    Trace(View& v) : m_view(v) {}
    Trace(View&& v) : m_view(v) {}

    Trace(Trace&& v) = default;
    Trace(const Trace& v) = default;

    value_type operator()() const {
        return reductions::CoefficientSum(
            unary::DiagonalView<const view_type>(m_view))();
    }

   private:
    View& m_view;
};

template <zipper::concepts::QualifiedViewDerived View>
Trace(View&) -> Trace<View>;

}  // namespace reductions
}  // namespace zipper::views
#endif
