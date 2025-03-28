#if !defined(ZIPPER_CONCEPTS_ZIPPERBASE_DERIVED_HPP)
#define ZIPPER_CONCEPTS_ZIPPERBASE_DERIVED_HPP
#include <type_traits>

#include "ViewDerived.hpp"

namespace zipper {
template <template <concepts::ViewDerived> typename DerivedT,
          concepts::ViewDerived View>
class ZIPPERBase;
}  // namespace zipper
namespace zipper::concepts {
namespace detail {
template <typename>
struct IsZIPPER : std::false_type {};

template <typename>
struct IsZIPPERBase : std::false_type {};
template <template <concepts::ViewDerived> typename DerivedT,
          concepts::ViewDerived View>
struct IsZIPPERBase<ZIPPERBase<DerivedT, View>> : std::true_type {};
}  // namespace detail

template <typename T>
concept ZIPPERBaseDerived = detail::IsZIPPERBase<T>::value;
}  // namespace zipper::concepts
#endif
