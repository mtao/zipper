#if !defined(ZIPPER_expression_TRACE_HPP)
#define ZIPPER_expression_TRACE_HPP

#include "CoefficientSum.hpp"
#include "ReductionBase.hpp"
#include "zipper/expression/unary/DiagonalExtract.hpp"

namespace zipper::expression {
namespace reductions {

template <typename Expr>
class Trace : public ReductionBase<Trace<Expr>, Expr> {
public:
  using Base = ReductionBase<Trace<Expr>, Expr>;
  using typename Base::expression_type;
  using typename Base::expression_traits;
  using typename Base::value_type;

  using Base::Base;
  using Base::expression;

  value_type operator()() const {
    return reductions::CoefficientSum(
        unary::DiagonalExtract<const expression_type&>(expression()))();
  }
};

template <zipper::concepts::QualifiedExpression E>
Trace(E &) -> Trace<E &>;

template <zipper::concepts::QualifiedExpression E>
Trace(E &&) -> Trace<E>;

} // namespace reductions
} // namespace zipper::expression
#endif
