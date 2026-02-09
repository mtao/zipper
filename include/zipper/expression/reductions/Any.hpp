#if !defined(ZIPPER_EXPRESSION_REDUCTIONS_ANY_HPP)
#define ZIPPER_EXPRESSION_REDUCTIONS_ANY_HPP

#include "zipper/concepts/Expression.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"
#include "zipper/utils/extents/all_extents_indices.hpp"

namespace zipper::expression::reductions {

template <zipper::concepts::QualifiedExpression Expression>
class Any {
public:
  using self_type = Any<Expression>;
  using expression_type = std::remove_reference_t<Expression>;
  using expression_traits =
      zipper::expression::detail::ExpressionTraits<expression_type>;
  using value_type = bool;

  Any(const expression_type &v) : m_expression(v) {}

  Any(Any &&v) = default;
  Any(const Any &v) = default;
  auto operator=(Any &&v) -> Any & = delete;
  auto operator=(const Any &v) -> Any & = delete;

  value_type operator()() const {
    for (const auto &i :
         zipper::utils::extents::all_extents_indices(m_expression.extents())) {
      if (value_type(std::apply(m_expression, i))) {
        return true;
      }
    }
    return false;
  }

private:
  const expression_type &m_expression;
};

template <zipper::concepts::QualifiedExpression Expression>
Any(const Expression &) -> Any<Expression>;

} // namespace zipper::expression::reductions
#endif
