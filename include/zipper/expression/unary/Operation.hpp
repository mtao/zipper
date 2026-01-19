#if !defined(ZIPPER_EXPRESSION_UNARY_OPERATION_HPP)
#define ZIPPER_EXPRESSION_UNARY_OPERATION_HPP

#include "UnaryExpressionBase.hpp"
#include "concepts/ScalarOperation.hpp"

namespace zipper::views {
namespace unary {
// TODO: operation should inherit std::unary_function<T,U>
template <zipper::concepts::Expression B,
          //
          // concepts::ScalarOperation<typename B::value_type> Operation,
          typename Op,
          //
          bool PreservesZeros = false>
// concepts::ScalarOperation<typename Child::value_type>
class Operation;

} // namespace unary
template <zipper::concepts::Expression Child,
          //
          // unary::concepts::ScalarOperation<typename Child::value_type>
          // Operation,
          typename Op,
          //
          bool PreservesZeros>
struct detail::ExpressionTraits<unary::Operation<Child, Op, PreservesZeros>>
    : public zipper::views::unary::detail::DefaultUnaryExpressionTraits<Child,
                                                                        false> {
  using child_traits = ExpressionTraits<Child>;
  using value_type = std::decay_t<decltype(std::declval<Operation>()(
      std::declval<typename child_traits::value_type>()))>;
};

// represents a coefficient-wise transformation of an underlyng view
namespace unary {

template <zipper::concepts::Expression Child,
          // concepts::ScalarOperation<typename Child::value_type> Operation,
          typename Operation,
          //
          bool PreservesZeros>

// requires(concepts::ScalarOperation<Child::value_type,Operation>)
class OperationExpression
    : public UnaryExpressionBase<
          OperationExpression<Child, Operation, PreservesZeros>, Child> {
public:
  using self_type = OperationExpression<Child, Operation, PreservesZeros>;
  using Base = UnaryExpressionBase<self_type, Child>;
  using traits = zipper::views::detail::ExpressionTraits<self_type>;
  constexpr static bool holds_extents = traits::holds_extents;
  static_assert(!holds_extents);
  using extents_type = traits::extents_type;
  using value_type = traits::value_type;

  using Base::extent;
  using Base::view;

  OperationExpression(Child &v, const Operation &op = {}) : Base(v), m_op(op) {}

  using child_value_type = traits::base_value_type;

  value_type get_value(const child_value_type &value) const {
    return value_type(m_op(value));
  }

private:
  Operation m_op;
};

template <typename A, zipper::concepts::Expression B>
OperationExpression(const A &a, B &b) -> OperationExpression<A, B>;
} // namespace unary
} // namespace zipper::views
#endif
