#if !defined(ZIPPER_EXPRESSION_REDUCTIONS_LPNORMPOWERED_HPP)
#define ZIPPER_EXPRESSION_REDUCTIONS_LPNORMPOWERED_HPP

#include "CoefficientSum.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"
#include "zipper/expression/unary/Abs.hpp"
#include "zipper/expression/unary/ScalarPower.hpp"
#include "zipper/expression/unary/PartialReduction.hpp"

namespace zipper::expression {
namespace reductions {
namespace detail {
template <index_type P>
  requires(P > 0)
struct lp_norm_powered_holder {
  template <zipper::concepts::QualifiedExpression Expr>
  class LpNormPowered {
  public:
    using self_type = LpNormPowered<Expr>;
    using expression_type = std::remove_reference_t<Expr>;
    using expression_traits =
        zipper::expression::detail::ExpressionTraits<expression_type>;
    using value_type = typename expression_traits::value_type;

    LpNormPowered(expression_type &v) : m_expression(v) {}

    LpNormPowered(LpNormPowered &&v) = default;
    LpNormPowered(const LpNormPowered &v) = default;

    value_type operator()() const {
      if constexpr (P % 2 == 0) {
        auto pow =
            unary::ScalarPower<const expression_type &, value_type>(m_expression, P);
        auto sum = reductions::CoefficientSum(pow);
        return sum();
      } else {
        auto abs = unary::Abs<const expression_type &>(m_expression);
        if constexpr (P == 1) {
          auto sum = reductions::CoefficientSum(abs);
          return sum();
        } else {
          auto pow =
              unary::ScalarPower<const expression_type &, value_type>(m_expression, P);
          auto sum = reductions::CoefficientSum(pow);
          return sum();
        }
      }
    }

  private:
    const expression_type &m_expression;
  };

  template <zipper::concepts::QualifiedExpression ExprType,
            rank_type... Indices>
  static auto reduction(std::remove_reference_t<ExprType> &expr) {
    return unary::PartialReduction<ExprType, LpNormPowered,
                                   Indices...>(expr);
  }
};
} // namespace detail

template <index_type P, zipper::concepts::QualifiedExpression Expr>
using LpNormPowered =
    typename detail::lp_norm_powered_holder<P>::template LpNormPowered<Expr>;

template <zipper::concepts::QualifiedExpression Expr>
using L2NormPowered =
    typename detail::lp_norm_powered_holder<2>::LpNormPowered<Expr>;
template <zipper::concepts::QualifiedExpression Expr>
using L1NormPowered =
    typename detail::lp_norm_powered_holder<1>::LpNormPowered<Expr>;

} // namespace reductions
} // namespace zipper::expression
#endif
