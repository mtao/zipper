#if !defined(ZIPPER_CONCEPTS_ZIPPERBASE_DERIVED_HPP)
#define ZIPPER_CONCEPTS_ZIPPERBASE_DERIVED_HPP
#include <type_traits>

#include "ViewDerived.hpp"

namespace zipper {
template <template <concepts::QualifiedViewDerived> typename DerivedT,
          concepts::QualifiedViewDerived View>
class ZipperBase;
} // namespace zipper
namespace zipper::concepts {
namespace detail {

template <typename> struct ZipperBaseDerived : public std::false_type {};
template <template <concepts::QualifiedViewDerived> typename DerivedT,
          concepts::QualifiedViewDerived View>
struct ZipperBaseDerived<ZipperBase<DerivedT, View>> : std::true_type {};

template <concepts::VectorBaseDerived View>
struct ZipperBaseDerived<View> : std::true_type {};
template <concepts::MatrixBaseDerived View>
struct ZipperBaseDerived<View> : std::true_type {};
template <concepts::TensorBaseDerived View>
struct ZipperBaseDerived<View> : std::true_type {};
template <concepts::FormBaseDerived View>
struct ZipperBaseDerived<View> : std::true_type {};

} // namespace detail

template <typename T>
concept ZipperBaseDerived = detail::ZipperBaseDerived<std::decay_t<T>>::value;
} // namespace zipper::concepts
#endif
