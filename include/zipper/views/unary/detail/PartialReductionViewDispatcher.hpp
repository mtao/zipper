#if !defined(ZIPPER_VIEWS_UNARY_DETAIL_PARTIALREDUCTIONVIEWDDISPATCHER_HPP)
#define ZIPPER_VIEWS_UNARY_DETAIL_PARTIALREDUCTIONVIEWDISPATCHER_HPP
#include "zipper/concepts/ViewDerived.hpp"
#include "zipper/views/reductions/CoefficientSum.hpp"
#include "zipper/views/reductions/LpNorm.hpp"
#include "zipper/views/unary/PartialReductionView.hpp"

namespace zipper::views::unary::detail {

template <zipper::concepts::QualifiedViewDerived ViewType, rank_type... Indices>
class PartialReductionViewDispatcher {
   public:
    PartialReductionViewDispatcher(ViewType& v) : m_view(v) {}
    auto sum() const {
        return views::unary::PartialReductionView<
            std::decay_t<ViewType>, views::reductions::CoefficientSum,
            Indices...>(m_view);
    }
    template <index_type P = 2>
    auto norm() const {
        using holder = views::reductions::detail::lp_norm_holder<P>;
        return holder::template reduction_view<std::decay_t<ViewType>,
                                               Indices...>(m_view);
    }
    template <index_type P = 2>
    auto norm_powered() const {
        using holder = views::reductions::detail::lp_norm_powered_holder<P>;
        return holder::template reduction_view<std::decay_t<ViewType>,
                                               Indices...>(m_view);
    }

   private:
    ViewType& m_view;
};

}  // namespace zipper::views::unary::detail

#endif
