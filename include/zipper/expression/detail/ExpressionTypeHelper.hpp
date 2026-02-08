#if !defined(ZIPPER_EXPRESSION_DETAIL_EXPRESSION_TYPE_HELPER_HPP)
#define ZIPPER_EXPRESSION_DETAIL_EXPRESSION_TYPE_HELPER_HPP
#include "zipper/concepts/Expression.hpp"
namespace zipper::expression::detail {
template <typename T>
struct ExpressionTypeHelper {
    using type = T::expression_type;
};

template <zipper::concepts::Expression T>
struct ExpressionTypeHelper<T> {
    using type = T;
};

}  // namespace zipper::expression::detail
#endif
