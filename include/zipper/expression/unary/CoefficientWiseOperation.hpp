#if !defined(ZIPPER_EXPRESSION_UNARY_COEFFICIENTWISE_OPERATION_HPP)
#define ZIPPER_EXPRESSION_UNARY_COEFFICIENTWISE_OPERATION_HPP

#include "UnaryExpressionBase.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"

namespace zipper::expression {
namespace unary {
template <zipper::concepts::Expression B, typename Op>
class CoefficientWiseOperation;

} // namespace unary
template <zipper::concepts::Expression Child, typename Op>
struct expression::detail::ExpressionTraits<
    unary::CoefficientWiseOperation<Child, Op>>
    : public zipper::expression::unary::detail::DefaultUnaryExpressionTraits<
          Child> {
  using child_traits = ExpressionTraits<Child>;
  using value_type = std::decay_t<decltype(std::declval<Op>()(
      std::declval<typename child_traits::value_type>()))>;
  // CoefficientWiseOperation computes values on the fly — not referrable or
  // assignable
  constexpr static zipper::detail::AccessFeatures access_features = {
      .is_const = true,
      .is_reference = false,
      .is_alias_free = child_traits::access_features.is_alias_free,
  };
  consteval static auto is_const_valued() -> bool {
    return access_features.is_const;
  }
  consteval static auto is_reference_valued() -> bool {
    return access_features.is_reference;
  }
  consteval static auto is_assignable() -> bool {
    return !is_const_valued() && is_reference_valued();
  }
  consteval static auto is_referrable() -> bool {
    return access_features.is_reference;
  }

  constexpr static bool is_writable = is_assignable();
};

// represents a coefficient-wise transformation of an underlyng expression
namespace unary {

template <zipper::concepts::Expression Child, typename Operation>
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

  CoefficientWiseOperation(std::remove_reference_t<Child> &v, Operation const &op = {})
      : Base(v), m_op(op) {}

  using child_value_type = traits::base_value_type;

  auto get_value(const child_value_type &value) const -> value_type {
    return value_type(m_op(value));
  }

private:
  Operation m_op;

};

template <typename A, zipper::concepts::Expression B>
CoefficientWiseOperation(const A &a, B &b) -> CoefficientWiseOperation<A, B>;
} // namespace unary
} // namespace zipper::expression
#endif
