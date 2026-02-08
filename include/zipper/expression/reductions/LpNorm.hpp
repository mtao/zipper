#if !defined(ZIPPER_EXPRESSION_REDUCTIONS_LPNORM_HPP)
#define ZIPPER_EXPRESSION_REDUCTIONS_LPNORM_HPP

#include <cmath>

#include "LpNormPowered.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"

namespace zipper::expression {
namespace reductions {

namespace detail {
template <index_type P>
struct lp_norm_holder {
  template <zipper::concepts::QualifiedExpression Expr>
  class LpNorm {
  public:
    using self_type = LpNorm<Expr>;
    using expression_type = Expr;
    using expression_traits =
        zipper::expression::detail::ExpressionTraits<expression_type>;
    using value_type = typename expression_traits::value_type;

    LpNorm(Expr &v) : m_expression(v) {}
    LpNorm(Expr &&v) : m_expression(v) {}

    LpNorm(LpNorm &&v) = default;
    LpNorm(const LpNorm &v) = default;

    value_type operator()() const {
      auto v = LpNormPowered<P, expression_type>(m_expression)();
      if constexpr (P == 1) {
        return v;
      } else if constexpr (P == 2) {
        return std::sqrt(v);
      } else {
        return std::pow(v, value_type(1.0) / P);
      }
    }

  private:
    Expr &m_expression;
  };
  template <zipper::concepts::QualifiedExpression ExprType,
            rank_type... Indices>
  static auto reduction_view(ExprType &expr) {
    return unary::PartialReduction<ExprType, LpNorm, Indices...>(expr);
  }
};
} // namespace detail

template <index_type P, zipper::concepts::QualifiedExpression Expr>
using LpNorm =
    typename detail::template lp_norm_holder<P>::template LpNorm<Expr>;

template <zipper::concepts::QualifiedExpression Expr>
using L2Norm = typename detail::lp_norm_holder<2>::LpNorm<Expr>;
template <zipper::concepts::QualifiedExpression Expr>
using L1Norm = typename detail::lp_norm_holder<1>::LpNorm<Expr>;

} // namespace reductions
} // namespace zipper::expression
#endif
