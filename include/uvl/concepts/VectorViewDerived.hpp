#if !defined(UVL_CONCEPTS_VECTOR_VIEW_DERIVED_HPP)
#define UVL_CONCEPTS_VECTOR_VIEW_DERIVED_HPP
#include "ViewDerived.hpp"
#include "TensorViewDerived.hpp"
namespace uvl::concepts {

template <typename T>
concept MatrixViewDerived = TensorViewDerived<T,1>;
}  // namespace uvl::concepts
#endif
