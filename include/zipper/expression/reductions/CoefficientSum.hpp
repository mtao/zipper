#if !defined(ZIPPER_expression_COEFFICIENTSUM_HPP)
#define ZIPPER_expression_COEFFICIENTSUM_HPP

#include "ReductionBase.hpp"
#include "zipper/utils/extents/all_extents_indices.hpp"

namespace zipper::expression::reductions {

template <typename Expr>
class CoefficientSum : public ReductionBase<CoefficientSum<Expr>, Expr> {
public:
  using Base = ReductionBase<CoefficientSum<Expr>, Expr>;
  using typename Base::expression_type;
  using typename Base::expression_traits;
  using typename Base::value_type;
  /// unqualified type of the underlying object
  using element_type = typename expression_traits::element_type;

  using Base::Base;  // inherit constructors
  using Base::expression;

  auto operator()() const -> element_type {
    element_type v = 0.0;
    for (const auto &i :
         zipper::utils::extents::all_extents_indices(expression().extents())) {
      v += std::apply(expression(), i);
    }
    return v;
  }
};

// Deduction guides: lvalue → ref storage, rvalue → value storage
template <zipper::concepts::QualifiedExpression E>
CoefficientSum(E &) -> CoefficientSum<E &>;

template <zipper::concepts::QualifiedExpression E>
CoefficientSum(E &&) -> CoefficientSum<E>;

} // namespace zipper::expression::reductions

#endif
