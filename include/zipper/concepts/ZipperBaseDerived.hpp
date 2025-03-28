#if !defined(ZIPPER_CONCEPTS_ZIPPERBASE_DERIVED_HPP)
#define ZIPPER_CONCEPTS_ZIPPERBASE_DERIVED_HPP
#include <type_traits>

#include "ViewDerived.hpp"

namespace zipper {
template <template <concepts::ViewDerived> typename DerivedT,
          concepts::ViewDerived View>
class ZipperBase;
}  // namespace zipper
namespace zipper::concepts {
namespace detail {
template <typename>
struct IsZIPPER : std::false_type {};

template <typename>
struct IsZipperBase : std::false_type {};
template <template <concepts::ViewDerived> typename DerivedT,
          concepts::ViewDerived View>
struct IsZipperBase<ZipperBase<DerivedT, View>> : std::true_type {};
}  // namespace detail

template <typename T>
concept ZipperBaseDerived = detail::IsZipperBase<T>::value;
}  // namespace zipper::concepts
#endif
