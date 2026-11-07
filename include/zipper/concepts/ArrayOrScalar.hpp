#if !defined(ZIPPER_CONCEPTS_ARRAYORSCALAR_HPP)
#define ZIPPER_CONCEPTS_ARRAYORSCALAR_HPP
#include "Array.hpp"
#include "Zipper.hpp"

#include <type_traits>

namespace zipper::concepts {

/// An array, or a scalar broadcast to an array's shape (as accepted by
/// select, clamp, ...).
template <typename T>
concept ArrayOrScalar =
    Array<T> || (!Zipper<T> && std::is_scalar_v<std::remove_cvref_t<T>>);

} // namespace zipper::concepts
#endif
