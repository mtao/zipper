#if !defined(ZIPPER_expression_UNARY_SCALAROPERATIONVIEW_HPP)
#define ZIPPER_expression_UNARY_SCALAROPERATIONVIEW_HPP

#include "UnaryExpressionBase.hpp"

namespace zipper::expression {
namespace unary {
template <zipper::concepts::Expression Child, typename Operation,
          typename Scalar, bool ScalarOnRight = false>
class ScalarOperation;

}
template <zipper::concepts::Expression Child, typename Operation,
          typename Scalar, bool ScalarOnRight>
struct detail::ExpressionTraits<
    unary::ScalarOperation<Child, Operation, Scalar, ScalarOnRight>>
    : public zipper::expression::unary::detail::DefaultUnaryExpressionTraits<
          Child> {
  using ChildTraits = ExpressionTraits<Child>;
  using value_type = decltype(std::declval<Operation>()(
      std::declval<typename ChildTraits::value_type>(),
      std::declval<Scalar>()));
  constexpr static bool is_writable = false;
};

namespace unary {
template <zipper::concepts::Expression Child, typename Operation,
          typename Scalar, bool ScalarOnRight>
class ScalarOperation
    : public UnaryExpressionBase<
          ScalarOperation<Child, Operation, Scalar, ScalarOnRight>, Child> {
public:
  using self_type = ScalarOperation<Child, Operation, Scalar, ScalarOnRight>;
  using traits = zipper::expression::detail::ExpressionTraits<self_type>;
  using extents_type = traits::extents_type;
  using value_type = traits::value_type;

  using Base = UnaryExpressionBase<self_type, Child>;
  using Base::view;

  ScalarOperation(Child &a, const Scalar &b, const Operation &op = {})
    requires(ScalarOnRight)
      : Base(a), m_op(op), m_scalar(b) {}
  ScalarOperation(const Scalar &a, Child &b, const Operation &op = {})
    requires(!ScalarOnRight)
      : Base(b), m_op(op), m_scalar(a) {}

  // using child_value_type = traits::base_value_type;

  auto get_value(const auto &value) const -> value_type {
    if constexpr (ScalarOnRight) {
      return m_op(value, m_scalar);
    } else {
      return m_op(m_scalar, value);
    }
  }

private:
  Operation m_op;
  Scalar m_scalar;
};

template <zipper::concepts::Expression Child, typename Operation,
          typename Scalar>
ScalarOperation(const Child &a, const Scalar &b, const Operation &op)
    -> ScalarOperation<Child, Scalar, Operation, true>;
template <zipper::concepts::Expression Child, typename Operation,
          typename Scalar>
ScalarOperation(const Scalar &a, const Child &b, const Operation &op)
    -> ScalarOperation<Child, Scalar, Operation, false>;
} // namespace unary
} // namespace zipper::expression
#endif
