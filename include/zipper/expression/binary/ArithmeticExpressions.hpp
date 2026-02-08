#if !defined(ZIPPER_EXPRESSION_BINARY_ARITHMETICEXPRESSIONS_HPP)
#define ZIPPER_EXPRESSION_BINARY_ARITHMETICEXPRESSIONS_HPP

#include <functional>

#include "Operation.hpp"
#include "detail/minmax.hpp"

namespace zipper::expression::binary {
namespace _detail_arith {
template <concepts::QualifiedExpression ExprA, concepts::QualifiedExpression ExprB>
struct scalar_type {
    using ATraits = expression::detail::ExpressionTraits<ExprA>;
    using BTraits = expression::detail::ExpressionTraits<ExprB>;

    using a_value_type = ATraits::value_type;
    using b_value_type = BTraits::value_type;
    static_assert(std::is_same_v<a_value_type, b_value_type>);
    using type = a_value_type;
};

template <zipper::concepts::QualifiedExpression ExprA, zipper::concepts::QualifiedExpression ExprB>
using scalar_type_t = scalar_type<ExprA, ExprB>::type;
}  // namespace _detail_arith

template <zipper::concepts::QualifiedExpression ExprA, zipper::concepts::QualifiedExpression ExprB>
using Plus =
    Operation<ExprA, ExprB, std::plus<_detail_arith::scalar_type_t<ExprA, ExprB>>>;

template <zipper::concepts::QualifiedExpression ExprA, zipper::concepts::QualifiedExpression ExprB>
using Minus =
    Operation<ExprA, ExprB,
                  std::minus<_detail_arith::scalar_type_t<ExprA, ExprB>>>;

template <zipper::concepts::QualifiedExpression ExprA, zipper::concepts::QualifiedExpression ExprB>
using Multiplies =
    Operation<ExprA, ExprB,
                  std::multiplies<_detail_arith::scalar_type_t<ExprA, ExprB>>>;

template <zipper::concepts::QualifiedExpression ExprA, zipper::concepts::QualifiedExpression ExprB>
using Divides =
    Operation<ExprA, ExprB,
                  std::divides<_detail_arith::scalar_type_t<ExprA, ExprB>>>;

template <zipper::concepts::QualifiedExpression ExprA, zipper::concepts::QualifiedExpression ExprB>
using Modulus =
    Operation<ExprA, ExprB,
                  std::modulus<_detail_arith::scalar_type_t<ExprA, ExprB>>>;

template <zipper::concepts::QualifiedExpression ExprA, zipper::concepts::QualifiedExpression ExprB>
using EqualsTo =
    Operation<ExprA, ExprB,
                  std::equal_to<_detail_arith::scalar_type_t<ExprA, ExprB>>>;

template <zipper::concepts::QualifiedExpression ExprA, zipper::concepts::QualifiedExpression ExprB>
using NotEqualsTo =
    Operation<ExprA, ExprB,
                  std::not_equal_to<_detail_arith::scalar_type_t<ExprA, ExprB>>>;

template <zipper::concepts::QualifiedExpression ExprA, zipper::concepts::QualifiedExpression ExprB>
using Greater =
    Operation<ExprA, ExprB,
                  std::greater<_detail_arith::scalar_type_t<ExprA, ExprB>>>;

template <zipper::concepts::QualifiedExpression ExprA, zipper::concepts::QualifiedExpression ExprB>
using Less =
    Operation<ExprA, ExprB, std::less<_detail_arith::scalar_type_t<ExprA, ExprB>>>;

template <zipper::concepts::QualifiedExpression ExprA, zipper::concepts::QualifiedExpression ExprB>
using GreaterEqual =
    Operation<ExprA, ExprB,
                  std::greater_equal<_detail_arith::scalar_type_t<ExprA, ExprB>>>;

template <zipper::concepts::QualifiedExpression ExprA, zipper::concepts::QualifiedExpression ExprB>
using LessEqual =
    Operation<ExprA, ExprB,
                  std::less_equal<_detail_arith::scalar_type_t<ExprA, ExprB>>>;

template <zipper::concepts::QualifiedExpression ExprA, zipper::concepts::QualifiedExpression ExprB>
using LogicalAnd =
    Operation<ExprA, ExprB,
                  std::logical_and<_detail_arith::scalar_type_t<ExprA, ExprB>>>;

template <zipper::concepts::QualifiedExpression ExprA, zipper::concepts::QualifiedExpression ExprB>
using LogicalOr =
    Operation<ExprA, ExprB,
                  std::logical_or<_detail_arith::scalar_type_t<ExprA, ExprB>>>;

template <zipper::concepts::QualifiedExpression ExprA, zipper::concepts::QualifiedExpression ExprB>
using BitAnd =
    Operation<ExprA, ExprB,
                  std::bit_and<_detail_arith::scalar_type_t<ExprA, ExprB>>>;

template <zipper::concepts::QualifiedExpression ExprA, zipper::concepts::QualifiedExpression ExprB>
using BitOr =
    Operation<ExprA, ExprB,
                  std::bit_or<_detail_arith::scalar_type_t<ExprA, ExprB>>>;

template <zipper::concepts::QualifiedExpression ExprA, zipper::concepts::QualifiedExpression ExprB>
using BitXor =
    Operation<ExprA, ExprB,
                  std::bit_xor<_detail_arith::scalar_type_t<ExprA, ExprB>>>;

template <zipper::concepts::QualifiedExpression ExprA, zipper::concepts::QualifiedExpression ExprB>
using Min = Operation<ExprA, ExprB,
                              detail::min<_detail_arith::scalar_type_t<ExprA, ExprB>>>;

template <zipper::concepts::QualifiedExpression ExprA, zipper::concepts::QualifiedExpression ExprB>
using Max = Operation<ExprA, ExprB,
                              detail::max<_detail_arith::scalar_type_t<ExprA, ExprB>>>;

}  // namespace zipper::expression::binary

#endif
