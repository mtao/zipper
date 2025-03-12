
#if !defined(UVL_CONCEPTS_LINEARVIEW_HPP)
#define UVL_CONCEPTS_LINEARVIEW_HPP
#include "MappedViewDerived.hpp"
#include "VectorViewDerived.hpp"

namespace uvl::concepts {

template <typename T>
concept LinearView = MappedViewDerived<T> || VectorViewDerived<T>;
}  // namespace uvl::concepts
#endif
