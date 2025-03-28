#if !defined(ZIPPER_CONCEPTS_MATRIX_VIEW_DERIVED_HPP)
#define ZIPPER_CONCEPTS_MATRIX_VIEW_DERIVED_HPP
#include <type_traits>

#include "TensorViewDerived.hpp"
namespace zipper::concepts {

template <typename T>
concept MatrixViewDerived = TensorViewDerived<T, 2>;
}  // namespace zipper::concepts
#endif
