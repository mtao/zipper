#if !defined(ZIPPER_CONCEPTS_ZIPPERBASE_DERIVED_HPP)
#define ZIPPER_CONCEPTS_ZIPPERBASE_DERIVED_HPP
#include "ArrayBaseDerived.hpp"
#include "FormBaseDerived.hpp"
#include "TensorBaseDerived.hpp"
#include "VectorBaseDerived.hpp"

namespace zipper::concepts {
template <typename T>
concept ZipperBaseDerived = ArrayBaseDerived<T> || FormBaseDerived<T> ||
                         TensorBaseDerived<T> || VectorBaseDerived<T>;
}  // namespace zipper::concepts
#endif
