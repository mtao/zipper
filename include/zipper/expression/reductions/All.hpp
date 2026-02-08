#if !defined(ZIPPER_EXPRESSION_REDUCTIONS_ALL_HPP)
#define ZIPPER_EXPRESSION_REDUCTIONS_ALL_HPP

#include "zipper/concepts/Expression.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"
#include "zipper/utils/extents/all_extents_indices.hpp"

namespace zipper::expression::reductions {

template <zipper::concepts::QualifiedExpression Expression>
class All {
public:
  using self_type = All<Expression>;
  using expression_type = Expression;
  using expression_traits =
      zipper::expression::detail::ExpressionTraits<expression_type>;
  using value_type = bool;

  All(Expression &v) : m_expression(v) {}
  All(Expression &&v) : m_expression(v) {}

  All(All &&v) = default;
  All(const All &v) = default;

  value_type operator()() const {
    for (const auto &i :
         zipper::utils::extents::all_extents_indices(m_expression.extents())) {
      if (!value_type(std::apply(m_expression, i))) {
        return false;
      }
    }
    return true;
  }

private:
  const Expression &m_expression;
};

template <zipper::concepts::QualifiedExpression Expression>
All(Expression &) -> All<Expression>;

} // namespace zipper::expression::reductions
#endif
