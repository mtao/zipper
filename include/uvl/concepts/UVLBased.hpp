#if !defined(UVL_CONCEPTS_UVLBASE_DERIVED_HPP)
#define UVL_CONCEPTS_UVLBASE_DERIVED_HPP
#include "ArrayBaseDerived.hpp"
#include "FormBaseDerived.hpp"
#include "TensorBaseDerived.hpp"
#include "VectorBaseDerived.hpp"

namespace uvl::concepts {
template <typename T>
concept UVLBaseDerived = ArrayBaseDerived<T> || FormBaseDerived<T> ||
                         TensorBaseDerived<T> || VectorBaseDerived<T>;
}  // namespace uvl::concepts
#endif
