
#if !defined(ZIPPER_EXPRESSION_CONCEPTS_REDUCTION_HPP)
#define ZIPPER_EXPRESSION_CONCEPTS_REDUCTION_HPP
#include <concepts>

#include "zipper/concepts/Expression.hpp"

namespace zipper::expression::concepts {
namespace detail {
template <typename T>
struct IsReduction : public std::false_type {};

template <zipper::concepts::Expression T>
    requires(T::extents_type::rank() == 0)
struct IsReduction<T> : public std::true_type {};
}  // namespace detail

template <typename T>
concept Reduction = detail::IsReduction<std::decay_t<T>>::value;
}  // namespace zipper::expression::concepts
#endif
