

#if !defined(ZIPPER_CONCEPTS_ARRAYBASE_DERIVED_HPP)
#define ZIPPER_CONCEPTS_ARRAYBASE_DERIVED_HPP
#include <concepts>
#include <type_traits>

#include "ExtentsType.hpp"
#include "ViewDerived.hpp"
#include "zipper/types.hpp"

namespace zipper {
template <concepts::ViewDerived T>
class ArrayBase;
template <typename ValueType, concepts::ExtentsType Extents,
          bool LeftMajor = false>
class Array_;
}  // namespace zipper
namespace zipper::concepts {
namespace detail {
template <typename>
struct IsArray : std::false_type {};
template <typename ValueType, concepts::ExtentsType Extents, bool LeftMajor>
struct IsArray<Array_<ValueType, Extents, LeftMajor>> : std::true_type {};

template <typename>
struct IsArrayBase : std::false_type {};
template <typename T>
struct IsArrayBase<ArrayBase<T>> : std::true_type {};
}  // namespace detail

template <typename T>
concept ArrayBaseDerived =
    (concepts::ViewDerived<T> && std::derived_from<T, zipper::ArrayBase<T>>) ||
    detail::IsArray<T>::value || detail::IsArrayBase<T>::value;
}  // namespace zipper::concepts
#endif
