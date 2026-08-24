#if !defined(ZIPPER_EXPRESSION_UNARY_ABS_HPP)
#define ZIPPER_EXPRESSION_UNARY_ABS_HPP

#include "CoefficientWiseOperation.hpp"
#include "detail/ZeroPreserving.hpp"
#include <zipper/utils/scalar_math.hpp>

namespace zipper::expression::unary {
namespace detail {
    template <typename A>
    struct abs {
        static constexpr A operator()(const A &a) {
            return utils::scalar_math::absolute_value(a);
        }
    };
} // namespace detail

template <zipper::concepts::Expression Child>
using Abs = CoefficientWiseOperation<
    Child,
    detail::abs<typename zipper::expression::detail::ExpressionTraits<
        Child>::value_type>>;

} // namespace zipper::expression::unary

template <typename T>
struct zipper::expression::detail::is_zero_preserving_unary_op<
    zipper::expression::unary::detail::abs<T>> : std::true_type {};

#endif
