#if !defined(ZIPPER_CONCEPTS_MATRIXBASEDERIVED_HPP)
#define ZIPPER_CONCEPTS_MATRIXBASEDERIVED_HPP
#include <type_traits>

#include "Expression.hpp"

namespace zipper {
template <concepts::Expression T> class MatrixBase;
template <typename T, index_type R> class Matrix;
} // namespace zipper
namespace zipper::concepts {
namespace detail {
template <typename> struct IsMatrixBaseDerived : std::false_type {};

} // namespace detail

template <typename T>
concept Matrix = detail::IsMatrixBaseDerived<std::decay_t<T>>::value;
template <typename T>
concept MatrixExpression = RankedExpression<T, 2>;
} // namespace zipper::concepts
#endif
