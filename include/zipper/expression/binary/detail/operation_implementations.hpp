#if !defined(ZIPPER_EXPRESSION_BINARY_DETAIL_SCALAR_BINARY_OPERATION_DECLARATION_HPP)

#define ZIPPER_EXPRESSION_BINARY_DETAIL_SCALAR_BINARY_OPERATION_DECLARATION_HPP

#include "zipper/concepts/Expression.hpp"

namespace zipper::expression::binary::detail {

/// Lvalue × lvalue: both children stored by const reference.
template <template <zipper::concepts::QualifiedExpression,
                    zipper::concepts::QualifiedExpression> class Op,
          template <typename> class BaseType,
          zipper::concepts::QualifiedExpression ExprA,
          zipper::concepts::QualifiedExpression ExprB>
auto operation_implementation(const ExprA& lhs, const ExprB& rhs) {
    using OpType = Op<const ExprA&, const ExprB&>;
    return BaseType<OpType>(std::in_place, lhs, rhs);
}

/// Rvalue × lvalue: LHS stored by value, RHS by const reference.
template <template <zipper::concepts::QualifiedExpression,
                    zipper::concepts::QualifiedExpression> class Op,
          template <typename> class BaseType,
          zipper::concepts::QualifiedExpression ExprA,
          zipper::concepts::QualifiedExpression ExprB>
    requires(!std::is_lvalue_reference_v<ExprA>)
auto operation_implementation(ExprA&& lhs, const ExprB& rhs) {
    using OpType = Op<std::decay_t<ExprA>, const ExprB&>;
    return BaseType<OpType>(std::in_place, std::move(lhs), rhs);
}

/// Lvalue × rvalue: LHS by const reference, RHS stored by value.
template <template <zipper::concepts::QualifiedExpression,
                    zipper::concepts::QualifiedExpression> class Op,
          template <typename> class BaseType,
          zipper::concepts::QualifiedExpression ExprA,
          zipper::concepts::QualifiedExpression ExprB>
    requires(!std::is_lvalue_reference_v<ExprB>)
auto operation_implementation(const ExprA& lhs, ExprB&& rhs) {
    using OpType = Op<const ExprA&, std::decay_t<ExprB>>;
    return BaseType<OpType>(std::in_place, lhs, std::move(rhs));
}

/// Rvalue × rvalue: both children stored by value.
template <template <zipper::concepts::QualifiedExpression,
                    zipper::concepts::QualifiedExpression> class Op,
          template <typename> class BaseType,
          zipper::concepts::QualifiedExpression ExprA,
          zipper::concepts::QualifiedExpression ExprB>
    requires(!std::is_lvalue_reference_v<ExprA> && !std::is_lvalue_reference_v<ExprB>)
auto operation_implementation(ExprA&& lhs, ExprB&& rhs) {
    using OpType = Op<std::decay_t<ExprA>, std::decay_t<ExprB>>;
    return BaseType<OpType>(std::in_place, std::move(lhs), std::move(rhs));
}

}  // namespace zipper::expression::binary::detail
#endif
