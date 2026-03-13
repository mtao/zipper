#if !defined(ZIPPER_EXPRESSION_REDUCTIONS_COEFFICIENTPRODUCT_HPP)
#define ZIPPER_EXPRESSION_REDUCTIONS_COEFFICIENTPRODUCT_HPP

#include "ReductionBase.hpp"
#include "zipper/utils/extents/all_extents_indices.hpp"

namespace zipper::expression::reductions {

template <typename Expr>
class CoefficientProduct
    : public ReductionBase<CoefficientProduct<Expr>, Expr> {
public:
  using Base = ReductionBase<CoefficientProduct<Expr>, Expr>;
  using typename Base::expression_type;
  using typename Base::expression_traits;
  using typename Base::value_type;

  using Base::Base;
  using Base::expression;

  value_type operator()() const {
    value_type v = 1.0;
    for (const auto &i :
         zipper::utils::extents::all_extents_indices(expression().extents())) {
      v *= std::apply(expression(), i);
    }
    return v;
  }
};

template <zipper::concepts::QualifiedExpression E>
CoefficientProduct(E &) -> CoefficientProduct<E &>;

template <zipper::concepts::QualifiedExpression E>
CoefficientProduct(E &&) -> CoefficientProduct<E>;

} // namespace zipper::expression::reductions
#endif
