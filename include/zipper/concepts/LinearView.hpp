
#if !defined(ZIPPER_CONCEPTS_LINEARVIEW_HPP)
#define ZIPPER_CONCEPTS_LINEARVIEW_HPP
#include "MappedViewDerived.hpp"
#include "VectorViewDerived.hpp"

namespace zipper::concepts {

template <typename T>
concept LinearView = MappedViewDerived<T> || VectorViewDerived<T>;
}  // namespace zipper::concepts
#endif
