
#if !defined(ZIPPER_CONCEPTS_MATRIXBASE_DERIVED_HPP)
#define ZIPPER_CONCEPTS_MATRIXBASE_DERIVED_HPP
#include <concepts>
#include <type_traits>

#include "MatrixViewDerived.hpp"

namespace zipper {
template <concepts::ViewDerived T>
class MatrixBase;
template <typename T, index_type R, index_type C>
class Matrix;
}  // namespace zipper
namespace zipper::concepts {
namespace detail {
template <typename>
struct IsMatrix : std::false_type {};
template <typename T, index_type R, index_type C>
struct IsMatrix<Matrix<T, R, C>> : std::true_type {};

template <typename>
struct IsMatrixBase : std::false_type {};
template <typename T>
struct IsMatrixBase<MatrixBase<T>> : std::true_type {};
}  // namespace detail

template <typename T>
concept MatrixBaseDerived =
    (concepts::MatrixViewDerived<T> &&
     std::derived_from<T, zipper::MatrixBase<T>>) ||
    detail::IsMatrix<T>::value || detail::IsMatrixBase<T>::value;
}  // namespace zipper::concepts
#endif
