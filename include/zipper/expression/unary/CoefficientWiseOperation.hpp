#if !defined(ZIPPER_EXPRESSION_UNARY_COEFFICIENTWISE_OPERATION_HPP)
#define ZIPPER_EXPRESSION_UNARY_COEFFICIENTWISE_OPERATION_HPP

#include "UnaryExpressionBase.hpp"
// TODO: perhaps switch back to using scalaroperation concept
// #include "concepts/ScalarOperation.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"

namespace zipper::expression {
namespace unary {
// TODO: operation should inherit std::unary_function<T,U>
template <zipper::concepts::Expression B,
          //
          // concepts::ScalarOperation<typename B::value_type> Operation,
          typename Op>
// concepts::ScalarOperation<typename Child::value_type>
class CoefficientWiseOperation;

} // namespace unary
template <zipper::concepts::Expression Child,
          //
          // unary::concepts::ScalarOperation<typename Child::value_type>
          // Operation,
          typename Op>
//
struct expression::detail::ExpressionTraits<
    unary::CoefficientWiseOperation<Child, Op>>
    : public zipper::expression::unary::detail::DefaultUnaryExpressionTraits<
          Child, false> {
  using child_traits = ExpressionTraits<Child>;
  using value_type = std::decay_t<decltype(std::declval<Op>()(
      std::declval<typename child_traits::value_type>()))>;
};

// represents a coefficient-wise transformation of an underlyng expression
namespace unary {

template <zipper::concepts::Expression Child,
          // concepts::ScalarOperation<typename Child::value_type> Operation,
          typename Operation>
//

// requires(concepts::ScalarOperation<Child::value_type,Operation>)
class CoefficientWiseOperation
    : public UnaryExpressionBase<CoefficientWiseOperation<Child, Operation>,
                                 Child> {
public:
  using self_type = CoefficientWiseOperation<Child, Operation>;
  using Base = UnaryExpressionBase<self_type, Child>;
  using traits = zipper::expression::detail::ExpressionTraits<self_type>;

  using extents_type = traits::extents_type;
  using value_type = traits::value_type;

  using Base::expression;
  using Base::extent;

  CoefficientWiseOperation(Child &v, Operation const &op = {})
      : Base(v), m_op(op) {}

  using child_value_type = traits::base_value_type;

  auto get_value(const child_value_type &value) const -> value_type {
    return value_type(m_op(value));
  }

private:
  Operation m_op;

  // debug typedef
  constexpr static bool holds_extents = traits::holds_extents;
  static_assert(!holds_extents);
};

template <typename A, zipper::concepts::Expression B>
CoefficientWiseOperation(const A &a, B &b) -> CoefficientWiseOperation<A, B>;
} // namespace unary
} // namespace zipper::expression
#endif
