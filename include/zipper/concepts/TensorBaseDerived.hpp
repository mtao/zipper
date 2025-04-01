#if !defined(ZIPPER_CONCEPTS_TENSORBASE_DERIVED_HPP)
#define ZIPPER_CONCEPTS_TENSORBASE_DERIVED_HPP
#include <concepts>
#include <type_traits>

#include "ViewDerived.hpp"
#include "zipper/types.hpp"

namespace zipper {
template <concepts::ViewDerived T>
class TensorBase;
// TODO: this really should get an extents back as a parameter
template <typename T, index_type... Indices>
class Tensor;
}  // namespace zipper
namespace zipper::concepts {
namespace detail {
template <typename>
struct IsTensor : std::false_type {};
template <typename T, index_type... Indices>
struct IsTensor<Tensor<T, Indices...>> : std::true_type {};

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
