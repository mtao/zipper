#if !defined(ZIPPER_CONCEPTS_ZIPPERBASE_DERIVED_HPP)
#define ZIPPER_CONCEPTS_ZIPPERBASE_DERIVED_HPP
#include <type_traits>

#include "ArrayBaseDerived.hpp"
#include "FormBaseDerived.hpp"
#include "MatrixBaseDerived.hpp"
#include "TensorBaseDerived.hpp"
#include "VectorBaseDerived.hpp"
#include "ViewDerived.hpp"

namespace zipper {
template <template <concepts::ViewDerived> typename DerivedT,
          concepts::ViewDerived View>
class ZipperBase;
}  // namespace zipper
namespace zipper::concepts {
namespace detail {

template <typename>
struct IsZipperBase : public std::false_type {};
template <template <concepts::ViewDerived> typename DerivedT,
          concepts::ViewDerived View>
struct IsZipperBase<ZipperBase<DerivedT, View>> : std::true_type {};

template <concepts::VectorBaseDerived View>
struct IsZipperBase<View> : std::true_type {};
template <concepts::ArrayBaseDerived View>
struct IsZipperBase<View> : std::true_type {};
template <concepts::MatrixBaseDerived View>
struct IsZipperBase<View> : std::true_type {};
template <concepts::TensorBaseDerived View>
struct IsZipperBase<View> : std::true_type {};
template <concepts::FormBaseDerived View>
struct IsZipperBase<View> : std::true_type {};

}  // namespace detail

template <typename T>
concept ZipperBaseDerived = detail::IsZipperBase<T>::value;
}  // namespace zipper::concepts
#endif
