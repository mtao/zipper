#if !defined(UVL_CONCEPTS_MATRIX_VIEW_DERIVED_HPP)
#define UVL_CONCEPTS_MATRIX_VIEW_DERIVED_HPP
#include <type_traits>

#include "TensorViewDerived.hpp"
namespace uvl::concepts {

template <typename T>
concept MatrixViewDerived = TensorViewDerived<T, 2>;
}  // namespace uvl::concepts
#endif
