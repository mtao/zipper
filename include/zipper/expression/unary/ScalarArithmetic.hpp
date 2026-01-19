#if !defined(ZIPPER_expression_UNARY_SCALARARITHMETICVIEW_HPP)
#define ZIPPER_expression_UNARY_SCALARARITHMETICVIEW_HPP

#include <functional>
#include <type_traits>

#include "CoefficientWiseOperation.hpp"
#include "ScalarOperation.hpp"

namespace zipper::expression::unary {

template <zipper::concepts::Expression Expression>
using NegateExpression =
    Operation<Expression,
              std::negate<typename zipper::expression::detail::ExpressionTraits<
                  Expression>::value_type>>;

template <zipper::concepts::Expression Expression>
using LogicalNotExpression =
    Operation<Expression,
              std::logical_not<typename zipper::expression::detail::
                                   ExpressionTraits<Expression>::value_type>>;

template <zipper::concepts::Expression Expression>
using BitNotExpression =
    Operation<Expression,
              std::bit_not<typename zipper::expression::detail::
                               ExpressionTraits<Expression>::value_type>>;

template <typename Scalar, zipper::concepts::Expression Expression,
          bool ScalarOnRight>
using ScalarPlusExpression = ScalarOperation<
    Expression,
    std::plus<std::common_type_t<Scalar, typename Expression::value_type>>,
    Scalar, ScalarOnRight>;

template <typename Scalar, zipper::concepts::Expression Expression,
          bool ScalarOnRight>
using ScalarMinusExpression = ScalarOperation<
    Expression,
    std::minus<std::common_type_t<Scalar, typename Expression::value_type>>,
    Scalar, ScalarOnRight>;

// coeff-wise product preserves zero-ness
template <typename Scalar, zipper::concepts::Expression Expression,
          bool ScalarOnRight>
using ScalarMultipliesExpression =
    ScalarOperation<Expression,
                    std::multiplies<std::common_type_t<
                        Scalar, typename Expression::value_type>>,
                    Scalar, ScalarOnRight, true>;

template <typename Scalar, zipper::concepts::Expression Expression,
          bool ScalarOnRight>
using ScalarDividesExpression = ScalarOperation<
    Expression,
    std::divides<std::common_type_t<Scalar, typename Expression::value_type>>,
    Scalar, ScalarOnRight>;

// modulus preserves zero if the modulus is on the right
template <typename Scalar, zipper::concepts::Expression Expression,
          bool ScalarOnRight>
using ScalarModulusExpression = ScalarOperation<
    Expression,
    std::modulus<std::common_type_t<Scalar, typename Expression::value_type>>,
    Scalar, ScalarOnRight, ScalarOnRight>;

template <typename Scalar, zipper::concepts::Expression Expression,
          bool ScalarOnRight>
using ScalarEqualsToExpression = ScalarOperation<
    Expression,
    std::equal_to<std::common_type_t<Scalar, typename Expression::value_type>>,
    Scalar, ScalarOnRight>;

template <typename Scalar, zipper::concepts::Expression Expression,
          bool ScalarOnRight>
using ScalarNotEqualsToExpression =
    ScalarOperation<Expression,
                    std::not_equal_to<std::common_type_t<
                        Scalar, typename Expression::value_type>>,
                    Scalar, ScalarOnRight>;

template <typename Scalar, zipper::concepts::Expression Expression,
          bool ScalarOnRight>
using ScalarGreaterExpression = ScalarOperation<
    Expression,
    std::greater<std::common_type_t<Scalar, typename Expression::value_type>>,
    Scalar, ScalarOnRight>;

template <typename Scalar, zipper::concepts::Expression Expression,
          bool ScalarOnRight>
using ScalarLessExpression = ScalarOperation<
    Expression,
    std::less<std::common_type_t<Scalar, typename Expression::value_type>>,
    Scalar, ScalarOnRight>;

template <typename Scalar, zipper::concepts::Expression Expression,
          bool ScalarOnRight>
using ScalarGreaterEqualExpression =
    ScalarOperation<Expression,
                    std::greater_equal<std::common_type_t<
                        Scalar, typename Expression::value_type>>,
                    Scalar, ScalarOnRight>;

template <typename Scalar, zipper::concepts::Expression Expression,
          bool ScalarOnRight>
using ScalarLessEqualExpression =
    ScalarOperation<Expression,
                    std::less_equal<std::common_type_t<
                        Scalar, typename Expression::value_type>>,
                    Scalar, ScalarOnRight>;

// logical and preserves false
template <typename Scalar, zipper::concepts::Expression Expression,
          bool ScalarOnRight>
using ScalarLogicalAndExpression =
    ScalarOperation<Expression,
                    std::logical_and<std::common_type_t<
                        Scalar, typename Expression::value_type>>,
                    Scalar, ScalarOnRight, true>;

template <typename Scalar, zipper::concepts::Expression Expression,
          bool ScalarOnRight>
using ScalarLogicalOrExpression =
    ScalarOperation<Expression,
                    std::logical_or<std::common_type_t<
                        Scalar, typename Expression::value_type>>,
                    Scalar, ScalarOnRight>;

template <typename Scalar, zipper::concepts::Expression Expression,
          bool ScalarOnRight>
using ScalarBitAndExpression = ScalarOperation<
    Expression,
    std::bit_and<std::common_type_t<Scalar, typename Expression::value_type>>,
    Scalar, ScalarOnRight>;

template <typename Scalar, zipper::concepts::Expression Expression,
          bool ScalarOnRight>
using ScalarBitOrExpression = ScalarOperation<
    Expression,
    std::bit_or<std::common_type_t<Scalar, typename Expression::value_type>>,
    Scalar, ScalarOnRight>;

template <typename Scalar, zipper::concepts::Expression Expression,
          bool ScalarOnRight>
using ScalarBitXorExpression = ScalarOperation<
    Expression,
    std::bit_xor<std::common_type_t<Scalar, typename Expression::value_type>>,
    Scalar, ScalarOnRight>;

} // namespace zipper::expression::unary

#endif
