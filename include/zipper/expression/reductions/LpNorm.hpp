#if !defined(ZIPPER_EXPRESSION_REDUCTIONS_LPNORM_HPP)
#define ZIPPER_EXPRESSION_REDUCTIONS_LPNORM_HPP

#include <cmath>

#include "LpNormPowered.hpp"
#include "ReductionBase.hpp"

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
      auto v = LpNormPowered<P, const expression_type &>(expression())();
      if constexpr (P == 1) {
        return v;
      } else if constexpr (P == 2) {
        return std::sqrt(v);
      } else {
        return std::pow(v, value_type(1.0) / P);
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
