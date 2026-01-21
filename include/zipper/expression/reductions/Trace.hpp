#if !defined(ZIPPER_expression_TRACE_HPP)
#define ZIPPER_expression_TRACE_HPP

#include "zipper/concepts/Expression.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"
#include "zipper/expression/reductions/CoefficientSum.hpp"
#include "zipper/expression/unary/Diagonal.hpp"

namespace zipper::expression {
namespace reductions {

template <zipper::concepts::QualifiedExpression Expression> class Trace {
public:
  using self_type = Trace<Expression>;
  using expression_type = Expression;
  using Expression_traits =
      zipper::expression::detail::ExpressionTraits<expression_type>;
  using value_type = typename Expression_traits::value_type;

  Trace(Expression &v) : m_Expression(v) {}
  Trace(Expression &&v) : m_Expression(v) {}

  Trace(Trace &&v) = default;
  Trace(const Trace &v) = default;

  value_type operator()() const {
    return reductions::CoefficientSum(
        unary::Diagonal<const expression_type>(m_Expression))();
  }

private:
  Expression &m_Expression;
};

template <zipper::concepts::QualifiedExpression Expression>
Trace(Expression &) -> Trace<Expression>;

} // namespace reductions
} // namespace zipper::expression
#endif
