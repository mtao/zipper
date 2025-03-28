#if !defined(ZIPPER_CONCEPTS_VECTOR_VIEW_DERIVED_HPP)
#define ZIPPER_CONCEPTS_VECTOR_VIEW_DERIVED_HPP
#include "ViewDerived.hpp"
#include "TensorViewDerived.hpp"
namespace zipper::concepts {

template <typename T>
concept VectorViewDerived = TensorViewDerived<T,1>;
}  // namespace zipper::concepts
#endif
