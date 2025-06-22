#if !defined(ZIPPER_CONCEPTS_TENSORBASE_DERIVED_HPP)
#define ZIPPER_CONCEPTS_TENSORBASE_DERIVED_HPP
#include <concepts>
#include <type_traits>

#include "ExtentsType.hpp"
#include "ViewDerived.hpp"
#include "zipper/types.hpp"

namespace zipper {
template <concepts::ViewDerived T>
class TensorBase;
template <typename ValueType, concepts::ExtentsType Extents,
          bool LeftMajor = true>
class Tensor_;
template <typename ValueType, concepts::ExtentsType Extents,
          bool LeftMajor = true>
class TensorSpan_;
}  // namespace zipper
namespace zipper::concepts {
namespace detail {
template <typename>
struct IsTensor : std::false_type {};
template <typename ValueType, concepts::ExtentsType Extents, bool LeftMajor>
struct IsTensor<Tensor_<ValueType, Extents, LeftMajor>> : std::true_type {};
template <typename ValueType, concepts::ExtentsType Extents, bool LeftMajor>
struct IsTensor<TensorSpan_<ValueType, Extents, LeftMajor>> : std::true_type {};

template <typename>
struct IsTensorBase : std::false_type {};
template <typename T>
struct IsTensorBase<TensorBase<T>> : std::true_type {};
}  // namespace detail

template <typename T>
concept TensorBaseDerived =
    (concepts::ViewDerived<T> && std::derived_from<T, zipper::TensorBase<T>>) ||
    detail::IsTensor<T>::value || detail::IsTensorBase<T>::value;
}  // namespace zipper::concepts
#endif
