#if !defined(ZIPPER_VIEWS_NORM_HPP)
#define ZIPPER_VIEWS_NORM_HPP

#include "LpNormPowered.hpp"
#include "zipper/concepts/ViewDerived.hpp"

namespace zipper::views {
namespace reductions {

namespace detail {
template <index_type P>
struct lp_norm_holder {
    template <zipper::concepts::ViewDerived View>
    class LpNorm {
       public:
        using self_type = LpNorm<View>;
        using view_type = View;
        using view_traits = zipper::views::detail::ViewTraits<view_type>;
        using value_type = typename View::value_type;

        LpNorm(View&& v) : m_view(v) {}
        LpNorm(const View& v) : m_view(v) {}

        LpNorm(LpNorm&& v) = default;
        LpNorm(const LpNorm& v) = default;

        value_type operator()() const {
            auto v = LpNormPowered<P, view_type>(m_view)();
            if constexpr (P == 1) {
                return v;
            } else if constexpr (P == 2) {
                return std::sqrt(v);
            } else {
                return std::pow(v, value_type(1.0) / P);
            }
        }

       private:
        const View& m_view;
    };
    template <zipper::concepts::ViewDerived ViewType,

              rank_type... Indices>
    static auto reduction_view(const ViewType& view) {
        return views::unary::PartialReductionView<std::decay_t<ViewType>,
                                                  LpNorm, Indices...>(view);
    }
};
}  // namespace detail

template <index_type P, zipper::concepts::ViewDerived View>
using LpNorm = typename detail::lp_norm_holder<P>::LpNorm<View>;

template <zipper::concepts::ViewDerived View>
using L2Norm = typename detail::lp_norm_holder<2>::LpNorm<View>;
template <zipper::concepts::ViewDerived View>
using L1Norm = typename detail::lp_norm_holder<1>::LpNorm<View>;

}  // namespace reductions
}  // namespace zipper::views
#endif
