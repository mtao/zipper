#if !defined(ZIPPER_EXPRESSION_UNARY_DETAIL_PARTIALREDUCTIONDISPATCHER_HPP)
#define ZIPPER_EXPRESSION_UNARY_DETAIL_PARTIALREDUCTIONDISPATCHER_HPP
#include "zipper/concepts/Expression.hpp"
#include "zipper/expression/reductions/CoefficientSum.hpp"
#include "zipper/expression/reductions/LpNorm.hpp"
#include "zipper/expression/unary/PartialReduction.hpp"
#include "zipper/expression/unary/PartialTransform.hpp"

namespace zipper::expression::unary::detail {

template <zipper::concepts::QualifiedExpression ExprType, rank_type... Indices>
class PartialReductionDispatcher {
  public:
    PartialReductionDispatcher(ExprType &v) : m_expression(v) {}
    auto &expression() const { return m_expression; }
    auto sum() const {
        return unary::PartialReduction<ExprType &,
                                       reductions::CoefficientSum,
                                       Indices...>(m_expression);
    }
    template <index_type P = 2>
    auto norm() const {
        using holder = reductions::detail::lp_norm_holder<P>;
        return holder::template reduction<ExprType &, Indices...>(m_expression);
    }
    template <index_type P = 2>
    auto norm_powered() const {
        using holder = reductions::detail::lp_norm_powered_holder<P>;
        return holder::template reduction<ExprType &, Indices...>(m_expression);
    }

    template <typename F>
    auto transform(F &&fn) const {
        return unary::PartialTransform<ExprType &, std::decay_t<F>, Indices...>(
            m_expression, std::forward<F>(fn));
    }

  private:
    ExprType &m_expression;
};

} // namespace zipper::expression::unary::detail

#endif
