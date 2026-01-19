#if !defined(ZIPPER_CONCEPTS_EXPRESSION_HPP)
#define ZIPPER_CONCEPTS_EXPRESSION_HPP
#include <concepts>

namespace zipper::expression {
template <typename T> class ExpressionBase;
}
namespace zipper::concepts {

/// Every view must be derived from either ExpressionBase via CRTP, with
/// constness qualification to specify if it is a const type view
template <typename T>
concept UnqualifiedExpression =
    std::derived_from<T, zipper::expression::ExpressionBase<T>> ||
    std::derived_from<T, zipper::expression::ExpressionBase<const T>>;

template <typename T>
concept Expression = UnqualifiedExpression<std::remove_cvref_t<T>>;

template <typename T>
concept QualifiedExpression = UnqualifiedExpression<std::remove_cvref_t<T>>;
} // namespace zipper::concepts
#endif
