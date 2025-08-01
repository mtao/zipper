#if !defined(ZIPPER_VIEWS_NORM_POWERED_HPP)
#define ZIPPER_VIEWS_NORM_POWERED_HPP

#include "zipper/concepts/ViewDerived.hpp"
#include "zipper/views/detail/ViewTraits.hpp"
#include "zipper/views/reductions/CoefficientSum.hpp"
#include "zipper/views/unary/AbsView.hpp"
#include "zipper/views/unary/PartialReductionView.hpp"
#include "zipper/views/unary/ScalarPowerView.hpp"

namespace zipper::views {
namespace reductions {
namespace detail {
template <index_type P>
    requires(P > 0)
struct lp_norm_powered_holder {
    template <zipper::concepts::QualifiedViewDerived View>
    class LpNormPowered {
       public:
        using self_type = LpNormPowered<View>;
        using view_type = View;
        using view_traits =
            zipper::views::detail::ViewTraits<std::decay_t<view_type>>;
        using value_type = typename view_traits::value_type;

        LpNormPowered(View& v) : m_view(v) {}
        LpNormPowered(View&& v) : m_view(v) {}

        LpNormPowered(LpNormPowered&& v) = default;
        LpNormPowered(const LpNormPowered& v) = default;

        value_type operator()() const {
            // return reductions::CoefficientSum(unary::DiagonalView(m_view))();
            if constexpr (P % 2 == 0) {
                auto pow = unary::ScalarPowerView<const view_type, value_type>(
                    m_view, P);
                auto sum = views::reductions::CoefficientSum(pow);
                return sum();
            } else {
                auto abs = views::unary::AbsView<const view_type>(m_view);
                if constexpr (P == 1) {
                    auto sum = views::reductions::CoefficientSum(abs);
                    return sum();
                } else {
                    auto pow =
                        unary::ScalarPowerView<const view_type, value_type>(
                            m_view, P);
                    auto sum = views::reductions::CoefficientSum(pow);
                    return sum();
                }
            }
        }

       private:
        const View& m_view;
    };

    template <zipper::concepts::QualifiedViewDerived ViewType,

              rank_type... Indices>
    static auto reduction_view(ViewType& view) {
        return views::unary::PartialReductionView<ViewType, LpNormPowered,
                                                  Indices...>(view);
    }
};
}  // namespace detail

template <index_type P, zipper::concepts::QualifiedViewDerived View>
using LpNormPowered =
    typename detail::lp_norm_powered_holder<P>::template LpNormPowered<View>;

template <zipper::concepts::QualifiedViewDerived View>
using L2NormPowered =
    typename detail::lp_norm_powered_holder<2>::LpNormPowered<View>;
template <zipper::concepts::QualifiedViewDerived View>
using L1NormPowered =
    typename detail::lp_norm_powered_holder<1>::LpNormPowered<View>;

}  // namespace reductions
}  // namespace zipper::views
#endif
