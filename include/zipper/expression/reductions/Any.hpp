#if !defined(ZIPPER_EXPRESSION_REDUCTIONS_ANY_HPP)
#define ZIPPER_EXPRESSION_REDUCTIONS_ANY_HPP

#include "ReductionBase.hpp"
#include "zipper/utils/extents/all_extents_indices.hpp"

namespace zipper::expression::reductions {

template <typename Expr>
class Any : public ReductionBase<Any<Expr>, Expr> {
public:
  using Base = ReductionBase<Any<Expr>, Expr>;
  using typename Base::expression_type;
  using typename Base::expression_traits;
  // Override value_type to bool
  using value_type = bool;

  using Base::Base;
  using Base::expression;

  value_type operator()() const {
    for (const auto &i :
         zipper::utils::extents::all_extents_indices(expression().extents())) {
      if (value_type(std::apply(expression(), i))) {
        return true;
      }
    }
    return false;
  }
};

template <zipper::concepts::QualifiedExpression E>
Any(E &) -> Any<E &>;

template <zipper::concepts::QualifiedExpression E>
Any(E &&) -> Any<E>;

} // namespace zipper::expression::reductions
#endif
