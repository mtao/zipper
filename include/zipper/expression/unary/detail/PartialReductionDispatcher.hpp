#if !defined(ZIPPER_EXPRESSION_UNARY_DETAIL_PARTIALREDUCTIONDISPATCHER_HPP)
#define ZIPPER_EXPRESSION_UNARY_DETAIL_PARTIALREDUCTIONDISPATCHER_HPP
#include "zipper/concepts/Expression.hpp"
#include "zipper/expression/reductions/CoefficientSum.hpp"
#include "zipper/expression/reductions/LpNorm.hpp"
#include "zipper/expression/unary/PartialReduction.hpp"

namespace zipper::expression::unary::detail {

template <zipper::concepts::QualifiedExpression ExprType, rank_type... Indices>
class PartialReductionDispatcher {
   public:
    PartialReductionDispatcher(std::remove_reference_t<ExprType>& v) : m_expression(v) {}
    auto sum() const {
        return unary::PartialReduction<
            ExprType &, reductions::CoefficientSum,
            Indices...>(m_expression);
    }
    template <index_type P = 2>
    auto norm() const {
        using holder = reductions::detail::lp_norm_holder<P>;
        return holder::template reduction<ExprType &,
                                               Indices...>(m_expression);
    }
    template <index_type P = 2>
    auto norm_powered() const {
        using holder = reductions::detail::lp_norm_powered_holder<P>;
        return holder::template reduction<ExprType &,
                                               Indices...>(m_expression);
    }

   private:
    std::remove_reference_t<ExprType>& m_expression;
};

}  // namespace zipper::expression::unary::detail

#endif
