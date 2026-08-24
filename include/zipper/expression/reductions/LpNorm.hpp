#if !defined(ZIPPER_EXPRESSION_REDUCTIONS_LPNORM_HPP)
#define ZIPPER_EXPRESSION_REDUCTIONS_LPNORM_HPP

#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

#include "LpNormPowered.hpp"
#include "ReductionBase.hpp"
#include "zipper/utils/extents/all_extents_indices.hpp"
#include "zipper/utils/scalar_math.hpp"

namespace zipper::expression {
namespace reductions {

namespace detail {
template <index_type P>
struct lp_norm_holder {
    template <typename Expr>
    class LpNorm : public ReductionBase<LpNorm<Expr>, Expr> {
      public:
        using Base = ReductionBase<LpNorm<Expr>, Expr>;
        using typename Base::expression_type;
        using typename Base::expression_traits;
        using typename Base::value_type;

        using Base::Base;
        using Base::expression;

        value_type operator()() const {
            if constexpr (P == 2 &&
                          utils::scalar_math::orthogonal_decomposition_scalar<
                              value_type>) {
                using accumulator_type = value_type;
                accumulator_type scale = accumulator_type{0};
                accumulator_type sumsq = accumulator_type{1};
                bool has_inf = false;

                for (const auto &i :
                     zipper::utils::extents::all_extents_indices(
                         expression().extents())) {
                    const accumulator_type value =
                        utils::scalar_math::absolute_value(
                            std::apply(expression(), i));
                    if constexpr (std::is_floating_point_v<value_type>) {
                        if (std::isnan(value)) {
                            return value_type(std::numeric_limits<
                                              accumulator_type>::quiet_NaN());
                        }
                        if (std::isinf(value)) {
                            has_inf = true;
                            continue;
                        }
                    }
                    if (value == accumulator_type{0}) { continue; }

                    if (scale < value) {
                        const accumulator_type ratio = scale / value;
                        sumsq = accumulator_type{1} + sumsq * ratio * ratio;
                        scale = value;
                    } else {
                        const accumulator_type ratio = value / scale;
                        sumsq += ratio * ratio;
                    }
                }

                if constexpr (std::is_floating_point_v<value_type>) {
                    if (has_inf) {
                        return value_type(std::numeric_limits<
                                          accumulator_type>::infinity());
                    }
                }
                if (scale == accumulator_type{0}) { return value_type{0}; }
                return scale * utils::scalar_math::square_root(sumsq);
            } else {
                auto v =
                    LpNormPowered<P, const expression_type &>(expression())();
                if constexpr (P == 1) {
                    return v;
                } else if constexpr (P == 2) {
                    using std::sqrt;
                    return value_type(sqrt(v));
                } else {
                    return utils::scalar_math::root(v, P);
                }
            }
        }
    };
    template <zipper::concepts::QualifiedExpression ExprType,
              rank_type... Indices>
    static auto reduction(ExprType &expr) {
        return unary::PartialReduction<ExprType, LpNorm, Indices...>(expr);
    }
};
} // namespace detail

template <index_type P, typename Expr>
using LpNorm =
    typename detail::template lp_norm_holder<P>::template LpNorm<Expr>;

template <typename Expr>
using L2Norm = typename detail::lp_norm_holder<2>::LpNorm<Expr>;
template <typename Expr>
using L1Norm = typename detail::lp_norm_holder<1>::LpNorm<Expr>;

} // namespace reductions
} // namespace zipper::expression
#endif
