#if !defined(ZIPPER_EXPRESSION_REDUCTIONS_LPNORMPOWERED_HPP)
#define ZIPPER_EXPRESSION_REDUCTIONS_LPNORMPOWERED_HPP

#include "CoefficientSum.hpp"
#include "ReductionBase.hpp"
#include "zipper/expression/unary/Abs.hpp"
#include "zipper/expression/unary/ScalarPower.hpp"
#include "zipper/expression/unary/PartialReduction.hpp"

namespace zipper::expression {
namespace reductions {
namespace detail {
template <index_type P>
  requires(P > 0)
struct lp_norm_powered_holder {
  template <typename Expr>
  class LpNormPowered
      : public ReductionBase<LpNormPowered<Expr>, Expr> {
  public:
    using Base = ReductionBase<LpNormPowered<Expr>, Expr>;
    using typename Base::expression_type;
    using typename Base::expression_traits;
    using typename Base::value_type;

    using Base::Base;
    using Base::expression;

    value_type operator()() const {
      if constexpr (P % 2 == 0) {
        auto pow =
            unary::ScalarPower<const expression_type&, value_type>(expression(), P);
        auto sum = reductions::CoefficientSum(pow);
        return sum();
      } else {
        auto abs = unary::Abs<const expression_type&>(expression());
        if constexpr (P == 1) {
          auto sum = reductions::CoefficientSum(abs);
          return sum();
        } else {
          auto pow =
              unary::ScalarPower<const expression_type&, value_type>(expression(), P);
          auto sum = reductions::CoefficientSum(pow);
          return sum();
        }
      }
    }
  };

  template <zipper::concepts::QualifiedExpression ExprType,
            rank_type... Indices>
  static auto reduction(ExprType &expr) {
    return unary::PartialReduction<ExprType, LpNormPowered,
                                   Indices...>(expr);
  }
};
} // namespace detail

template <index_type P, typename Expr>
using LpNormPowered =
    typename detail::lp_norm_powered_holder<P>::template LpNormPowered<Expr>;

template <typename Expr>
using L2NormPowered =
    typename detail::lp_norm_powered_holder<2>::LpNormPowered<Expr>;
template <typename Expr>
using L1NormPowered =
    typename detail::lp_norm_powered_holder<1>::LpNormPowered<Expr>;

} // namespace reductions
} // namespace zipper::expression
#endif
