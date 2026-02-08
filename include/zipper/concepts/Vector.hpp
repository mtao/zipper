#if !defined(ZIPPER_CONCEPTS_VECTOR_HPP)
#define ZIPPER_CONCEPTS_VECTOR_HPP
#include <concepts>
#include <type_traits>

#include "Expression.hpp"

namespace zipper {
template <concepts::Expression T> class VectorBase;
template <typename T, index_type R> class Vector;
} // namespace zipper
namespace zipper::concepts {
namespace detail {
template <typename> struct IsVector : std::false_type {};

} // namespace detail

template <typename T>
concept Vector = detail::IsVector<std::decay_t<T>>::value;
template <typename T>
concept VectorExpression = RankedExpression<T, 1>;

} // namespace zipper::concepts
#endif
