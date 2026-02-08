#if !defined(ZIPPER_CONCEPTS_DETAIL_IS_ZIPPER_BASE_HPP)
#define ZIPPER_CONCEPTS_DETAIL_IS_ZIPPER_BASE_HPP
#include <type_traits>

namespace zipper::concepts::detail {

// Default trait: most types are not ZipperBase-derived.
// Specialize this in the relevant concept header (e.g. Matrix.hpp,
// Vector.hpp) to register new ZipperBase families.
template <typename>
struct IsZipperBase : std::false_type {};

} // namespace zipper::concepts::detail
#endif
