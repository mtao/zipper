#if !defined(ZIPPER_EXPRESSION_BINARY_DETAIL_SCALAR_BINARY_OPERATION_DECLARATION_HPP)

#define ZIPPER_EXPRESSION_BINARY_DETAIL_SCALAR_BINARY_OPERATION_DECLARATION_HPP

#include "zipper/concepts/Expression.hpp"

namespace zipper::expression::binary::detail {
template <template <zipper::concepts::QualifiedExpression,
                    zipper::concepts::QualifiedExpression> class Op,
          template <typename> class BaseType,
          zipper::concepts::QualifiedExpression ExprA,
          zipper::concepts::QualifiedExpression ExprB>
auto operation_implementation(const ExprA& lhs, const ExprB& rhs) {
    using OpType = Op<const ExprA&, const ExprB&>;
    // Construct the expression in-place inside BaseType by forwarding
    // the constructor arguments, avoiding a move of the expression node.
    return BaseType<OpType>(std::in_place, lhs, rhs);
}

}  // namespace zipper::expression::binary::detail
#endif
