#if !defined(ZIPPER_EXPRESSION_REDUCTIONS_LPNORM_HPP)
#define ZIPPER_EXPRESSION_REDUCTIONS_LPNORM_HPP

#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

#include "LpNormPowered.hpp"
#include "ReductionBase.hpp"
#include "zipper/static_scalar.hpp"
#include "zipper/utils/extents/all_extents_indices.hpp"
#include "zipper/utils/scalar_math.hpp"

namespace zipper::expression {
namespace reductions {

template <typename P, typename Expr>
class LpNorm;

namespace detail {
template <index_type P>
struct lp_norm_holder {
    template <typename Expr>
    using LpNorm = reductions::LpNorm<std::integral_constant<index_type, P>, Expr>;
};

template <typename P, typename T>
concept NormOrder =
    (::zipper::concepts::StaticScalar<P>
     && std::convertible_to<typename std::remove_cvref_t<P>::value_type, T>)
    || std::convertible_to<P, T>;

template <typename P>
struct lp_norm_binder {
    template <typename Expr>
        requires NormOrder<P, typename std::remove_cvref_t<Expr>::value_type>
    using reduction = reductions::LpNorm<P, Expr>;
};
} // namespace detail

/// Lp norm reduction. `P` is either a static scalar (for compile-time
/// `norm<P>()`) or a stored scalar (for runtime `norm(p)`).
template <typename P, typename Expr>
class LpNorm : public ReductionBase<LpNorm<Expr, P>, Expr> {
      public:
    using Base = ReductionBase<LpNorm<Expr, P>, Expr>;
    using typename Base::expression_type;
    using typename Base::expression_traits;
    using typename Base::value_type;
    using Base::expression;

    explicit LpNorm(const Expr &expr, P p = {}) : Base(expr), m_p(std::move(p)) {}

    auto operator()() const -> value_type {
        if constexpr (::zipper::concepts::StaticScalar<P>) {
            if constexpr (P::value == 2
                          && utils::scalar_math::orthogonal_decomposition_scalar<value_type>) {
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
            constexpr auto p = P::value;
            auto v = LpNormPowered<p, const expression_type &>(expression())();
            if constexpr (p == 1) {
                return v;
            } else if constexpr (p == 2) {
                using std::sqrt;
                return value_type(sqrt(v));
            } else {
                return utils::scalar_math::root(v, p);
            }
            }
        } else {
            value_type sum = 0;
            for (const auto &index : zipper::utils::extents::all_extents_indices(expression().extents())) {
                using std::abs;
                using std::pow;
                sum += pow(abs(std::apply(expression(), index)), m_p);
            }
            using std::pow;
            return pow(sum, value_type(1) / m_p);
        }
    }

  private:
    P m_p;
};

template <typename Expr>
using L2Norm = LpNorm<std::integral_constant<index_type, 2>, Expr>;
template <typename Expr>
using L1Norm = LpNorm<std::integral_constant<index_type, 1>, Expr>;

} // namespace reductions
} // namespace zipper::expression
#endif
