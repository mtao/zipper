#if !defined(ZIPPER_expression_UNARY_SCALARARITHMETICVIEW_HPP)
#define ZIPPER_expression_UNARY_SCALARARITHMETICVIEW_HPP

#include <compare>
#include <functional>
#include <type_traits>

#include "CoefficientWiseOperation.hpp"
#include "ScalarOperation.hpp"

namespace zipper::expression::unary {

template <zipper::concepts::QualifiedExpression Expression>
using Negate = CoefficientWiseOperation<
    Expression,
    std::negate<typename zipper::expression::detail::ExpressionTraits<
        std::decay_t<Expression>>::value_type>>;

template <zipper::concepts::QualifiedExpression Expression>
using LogicalNot = CoefficientWiseOperation<
    Expression,
    std::logical_not<typename zipper::expression::detail::ExpressionTraits<
        std::decay_t<Expression>>::value_type>>;

template <zipper::concepts::QualifiedExpression Expression>
using BitNot = CoefficientWiseOperation<
    Expression,
    std::bit_not<typename zipper::expression::detail::ExpressionTraits<
        std::decay_t<Expression>>::value_type>>;

template <typename Scalar,
          zipper::concepts::QualifiedExpression Expression,
          bool ScalarOnRight>
using ScalarPlus =
    ScalarOperation<Expression,
                    std::plus<std::common_type_t<
                        Scalar,
                        typename std::decay_t<Expression>::value_type>>,
                    Scalar,
                    ScalarOnRight>;

template <typename Scalar,
          zipper::concepts::QualifiedExpression Expression,
          bool ScalarOnRight>
using ScalarMinus =
    ScalarOperation<Expression,
                    std::minus<std::common_type_t<
                        Scalar,
                        typename std::decay_t<Expression>::value_type>>,
                    Scalar,
                    ScalarOnRight>;

// coeff-wise product preserves zero-ness
template <typename Scalar,
          zipper::concepts::QualifiedExpression Expression,
          bool ScalarOnRight>
using ScalarMultiplies =
    ScalarOperation<Expression,
                    std::multiplies<std::common_type_t<
                        Scalar,
                        typename std::decay_t<Expression>::value_type>>,
                    Scalar,
                    ScalarOnRight>;

template <typename Scalar,
          zipper::concepts::QualifiedExpression Expression,
          bool ScalarOnRight>
using ScalarDivides =
    ScalarOperation<Expression,
                    std::divides<std::common_type_t<
                        Scalar,
                        typename std::decay_t<Expression>::value_type>>,
                    Scalar,
                    ScalarOnRight>;

// modulus preserves zero if the modulus is on the right
template <typename Scalar,
          zipper::concepts::QualifiedExpression Expression,
          bool ScalarOnRight>
using ScalarModulus =
    ScalarOperation<Expression,
                    std::modulus<std::common_type_t<
                        Scalar,
                        typename std::decay_t<Expression>::value_type>>,
                    Scalar,
                    ScalarOnRight>;

template <typename Scalar,
          zipper::concepts::QualifiedExpression Expression,
          bool ScalarOnRight>
using ScalarEqualsTo =
    ScalarOperation<Expression,
                    std::equal_to<std::common_type_t<
                        Scalar,
                        typename std::decay_t<Expression>::value_type>>,
                    Scalar,
                    ScalarOnRight>;

template <typename Scalar,
          zipper::concepts::QualifiedExpression Expression,
          bool ScalarOnRight>
using ScalarNotEqualsTo =
    ScalarOperation<Expression,
                    std::not_equal_to<std::common_type_t<
                        Scalar,
                        typename std::decay_t<Expression>::value_type>>,
                    Scalar,
                    ScalarOnRight>;

template <typename Scalar,
          zipper::concepts::QualifiedExpression Expression,
          bool ScalarOnRight>
using ScalarGreater =
    ScalarOperation<Expression,
                    std::greater<std::common_type_t<
                        Scalar,
                        typename std::decay_t<Expression>::value_type>>,
                    Scalar,
                    ScalarOnRight>;

template <typename Scalar,
          zipper::concepts::QualifiedExpression Expression,
          bool ScalarOnRight>
using ScalarLess =
    ScalarOperation<Expression,
                    std::less<std::common_type_t<
                        Scalar,
                        typename std::decay_t<Expression>::value_type>>,
                    Scalar,
                    ScalarOnRight>;

template <typename Scalar,
          zipper::concepts::QualifiedExpression Expression,
          bool ScalarOnRight>
using ScalarGreaterEqual =
    ScalarOperation<Expression,
                    std::greater_equal<std::common_type_t<
                        Scalar,
                        typename std::decay_t<Expression>::value_type>>,
                    Scalar,
                    ScalarOnRight>;

template <typename Scalar,
          zipper::concepts::QualifiedExpression Expression,
          bool ScalarOnRight>
using ScalarLessEqual =
    ScalarOperation<Expression,
                    std::less_equal<std::common_type_t<
                        Scalar,
                        typename std::decay_t<Expression>::value_type>>,
                    Scalar,
                    ScalarOnRight>;

// logical and preserves false
template <typename Scalar,
          zipper::concepts::QualifiedExpression Expression,
          bool ScalarOnRight>
using ScalarLogicalAnd =
    ScalarOperation<Expression,
                    std::logical_and<std::common_type_t<
                        Scalar,
                        typename std::decay_t<Expression>::value_type>>,
                    Scalar,
                    ScalarOnRight>;

template <typename Scalar,
          zipper::concepts::QualifiedExpression Expression,
          bool ScalarOnRight>
using ScalarLogicalOr =
    ScalarOperation<Expression,
                    std::logical_or<std::common_type_t<
                        Scalar,
                        typename std::decay_t<Expression>::value_type>>,
                    Scalar,
                    ScalarOnRight>;

template <typename Scalar,
          zipper::concepts::QualifiedExpression Expression,
          bool ScalarOnRight>
using ScalarBitAnd =
    ScalarOperation<Expression,
                    std::bit_and<std::common_type_t<
                        Scalar,
                        typename std::decay_t<Expression>::value_type>>,
                    Scalar,
                    ScalarOnRight>;

template <typename Scalar,
          zipper::concepts::QualifiedExpression Expression,
          bool ScalarOnRight>
using ScalarBitOr =
    ScalarOperation<Expression,
                    std::bit_or<std::common_type_t<
                        Scalar,
                        typename std::decay_t<Expression>::value_type>>,
                    Scalar,
                    ScalarOnRight>;

template <typename Scalar,
          zipper::concepts::QualifiedExpression Expression,
          bool ScalarOnRight>
using ScalarBitXor =
    ScalarOperation<Expression,
                    std::bit_xor<std::common_type_t<
                        Scalar,
                        typename std::decay_t<Expression>::value_type>>,
                    Scalar,
                    ScalarOnRight>;

// Element-wise three-way comparison against a scalar.
// Uses std::compare_three_way (transparent comparator), so the return type
// is deduced from the operands (e.g. std::partial_ordering for doubles,
// std::strong_ordering for ints).
template <typename Scalar,
          zipper::concepts::QualifiedExpression Expression,
          bool ScalarOnRight>
using ScalarThreeWayCompare =
    ScalarOperation<Expression, std::compare_three_way, Scalar, ScalarOnRight>;

} // namespace zipper::expression::unary

#endif
