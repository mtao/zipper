#if !defined(ZIPPER_EXPRESSION_UNARY_MATH_SQRT_HPP)
#define ZIPPER_EXPRESSION_UNARY_MATH_SQRT_HPP

#include "../CoefficientWiseOperation.hpp"
#include "../detail/ZeroPreserving.hpp"
#include <zipper/utils/scalar_math.hpp>

namespace zipper::expression::unary {
namespace detail {
    template <typename A>
    struct sqrt {
        static constexpr A operator()(const A &a) {
            return utils::scalar_math::square_root(a);
        }
    };
} // namespace detail

template <zipper::concepts::Expression Child>
using Sqrt = CoefficientWiseOperation<
    Child,
    detail::sqrt<typename zipper::expression::detail::ExpressionTraits<
        Child>::value_type>>;

} // namespace zipper::expression::unary

template <typename T>
struct zipper::expression::detail::is_zero_preserving_unary_op<
    zipper::expression::unary::detail::sqrt<T>> : std::true_type {};

#endif
