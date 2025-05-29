#if !defined(ZIPPER_CONCEPTS_ZIPPERBASE_DERIVED_HPP)
#define ZIPPER_CONCEPTS_ZIPPERBASE_DERIVED_HPP
#include <type_traits>

#include "ViewDerived.hpp"

namespace zipper {
template <template <concepts::ViewDerived> typename DerivedT,
          concepts::ViewDerived View>
class ZipperBase;
template <concepts::ViewDerived View>
class ArrayBase;
template <concepts::ViewDerived View>
class VectorBase;
template <concepts::ViewDerived View>
class MatrixBase;
template <concepts::ViewDerived View>
class FormBase;
template <concepts::ViewDerived View>
class TensorBase;
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

template <concepts::ViewDerived View>
struct IsZipperBase<ArrayBase<View>> : std::true_type {};
template <concepts::ViewDerived View>
struct IsZipperBase<VectorBase<View>> : std::true_type {};
template <concepts::ViewDerived View>
struct IsZipperBase<MatrixBase<View>> : std::true_type {};
template <concepts::ViewDerived View>
struct IsZipperBase<TensorBase<View>> : std::true_type {};
template <concepts::ViewDerived View>
struct IsZipperBase<FormBase<View>> : std::true_type {};

}  // namespace detail

template <typename T>
concept ZipperBaseDerived = detail::IsZipperBase<T>::value;
}  // namespace zipper::concepts
#endif
