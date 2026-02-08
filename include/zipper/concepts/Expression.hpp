#if !defined(ZIPPER_CONCEPTS_EXPRESSION_HPP)
#define ZIPPER_CONCEPTS_EXPRESSION_HPP
#include "zipper/types.hpp"
#include <concepts>

namespace zipper::expression {
template <typename T> class ExpressionBase;
}
namespace zipper::concepts {

/// Every expression must be derived from ExpressionBase via CRTP, with
/// constness qualification to specify if it is a const expression
template <typename T>
concept UnqualifiedExpression =
    std::derived_from<T, zipper::expression::ExpressionBase<T>> ||
    std::derived_from<T, zipper::expression::ExpressionBase<const T>>;

template <typename T>
concept Expression = UnqualifiedExpression<std::remove_cvref_t<T>>;

template <typename T>
concept QualifiedExpression = UnqualifiedExpression<std::remove_cvref_t<T>>;

/// Specifies something is a tensor with a particular rank
template <typename T, zipper::rank_type rank>
concept RankedExpression = Expression<T> && T::extents_type::rank() == rank;
} // namespace zipper::concepts
#endif
