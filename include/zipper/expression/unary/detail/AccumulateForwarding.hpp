#if !defined(ZIPPER_EXPRESSION_UNARY_DETAIL_ACCUMULATEFORWARDING_HPP)
#define ZIPPER_EXPRESSION_UNARY_DETAIL_ACCUMULATEFORWARDING_HPP

/// @file AccumulateForwarding.hpp
/// @brief Lets linear scalings of an accumulable expression (`s * X`,
/// `X * s`, `X / s`, `-X`) forward the accumulate protocol
/// (AccumulateAssignStrategy) to X with the scale folded into alpha, so e.g.
/// `C = 2 * (A * B)` reaches the GEMM kernel instead of the coefficient path.

#include <functional>
#include <type_traits>

#include "zipper/expression/detail/AssignStrategy.hpp"

namespace zipper::expression::unary::detail {

template <typename Op, template <typename> typename Tmpl>
inline constexpr bool is_op_v = false;
template <typename T, template <typename> typename Tmpl>
inline constexpr bool is_op_v<Tmpl<T>, Tmpl> = true;

/// `scalar (op) x` / `x (op) scalar` is linear in x.
template <typename Op, bool ScalarOnRight>
inline constexpr bool is_linear_scaling_v =
    is_op_v<Op, std::multiplies> || (ScalarOnRight && is_op_v<Op, std::divides>);

/// Strategy for a linear scaling of Child: accumulate when Child does.
template <typename Child, bool Linear>
using forwarded_assign_strategy_t = std::conditional_t<
    Linear && zipper::expression::detail::HasAccumulateStrategy<Child>,
    zipper::expression::detail::AccumulateAssignStrategy,
    zipper::expression::detail::DefaultAssignStrategy>;

} // namespace zipper::expression::unary::detail

#endif
