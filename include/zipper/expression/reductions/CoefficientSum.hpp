#if !defined(ZIPPER_expression_COEFFICIENTSUM_HPP)
#define ZIPPER_expression_COEFFICIENTSUM_HPP

#include "zipper/concepts/Expression.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"
#include "zipper/utils/extents/all_extents_indices.hpp"

namespace zipper::expression::reductions {

template <zipper::concepts::QualifiedExpression Expression>
class CoefficientSum {
public:
  using self_type = CoefficientSum<Expression>;
  using expression_type = Expression;
  using expression_traits =
      zipper::expression::detail::ExpressionTraits<expression_type>;
  using value_type = typename expression_traits::value_type;
  /// unqualified type of the underlying object
  using element_type = typename expression_traits::element_type;

  CoefficientSum(Expression &v) : m_expression(v) {}
  CoefficientSum(Expression &&v) : m_expression(v) {}

  CoefficientSum(CoefficientSum &&v) = default;
  CoefficientSum(const CoefficientSum &v) = default;
  auto operator=(CoefficientSum &&v) -> CoefficientSum & = delete;
  auto operator=(const CoefficientSum &v) -> CoefficientSum & = delete;

  auto operator()() const -> element_type {
    element_type v = 0.0;
    for (const auto &i :
         zipper::utils::extents::all_extents_indices(m_expression.extents())) {
      v += std::apply(m_expression, i);
    }
    return v;
  }

private:
  const Expression &m_expression;
}; // namespace unarytemplate<typenameA,typenameB>class AdditionExpression

template <zipper::concepts::QualifiedExpression Expression>
CoefficientSum(Expression &) -> CoefficientSum<Expression>;

} // namespace zipper::expression::reductions

#endif
