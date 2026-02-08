#if !defined(ZIPPER_EXPRESSION_REDUCTIONS_COEFFICIENTPRODUCT_HPP)
#define ZIPPER_EXPRESSION_REDUCTIONS_COEFFICIENTPRODUCT_HPP

#include "zipper/concepts/Expression.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"
#include "zipper/utils/extents/all_extents_indices.hpp"

namespace zipper::expression::reductions {

template <zipper::concepts::QualifiedExpression Expression>
class CoefficientProduct {
public:
  using self_type = CoefficientProduct<Expression>;
  using expression_type = Expression;
  using expression_traits =
      zipper::expression::detail::ExpressionTraits<expression_type>;
  using value_type = typename Expression::value_type;

  CoefficientProduct(Expression &v) : m_expression(v) {}
  CoefficientProduct(Expression &&v) : m_expression(v) {}

  CoefficientProduct(CoefficientProduct &&v) = default;
  CoefficientProduct(const CoefficientProduct &v) = default;

  value_type operator()() const {
    value_type v = 1.0;
    for (const auto &i :
         zipper::utils::extents::all_extents_indices(m_expression.extents())) {
      v *= std::apply(m_expression, i);
    }
    return v;
  }

private:
  const Expression &m_expression;
};

template <zipper::concepts::QualifiedExpression Expression>
CoefficientProduct(Expression &) -> CoefficientProduct<Expression>;

} // namespace zipper::expression::reductions
#endif
