#if !defined(UVL_CONCEPTS_TENSORBASE_DERIVED_HPP)
#define UVL_CONCEPTS_TENSORBASE_DERIVED_HPP
#include <concepts>
#include <type_traits>

#include "ViewDerived.hpp"
#include "uvl/types.hpp"

namespace uvl {
template <concepts::ViewDerived T>
class TensorBase;
template <typename T, index_type... Indices>
class Tensor;
}  // namespace uvl
namespace uvl::concepts {
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
    (concepts::ViewDerived<T> && std::derived_from<T, uvl::TensorBase<T>>) ||
    detail::IsTensor<T>::value || detail::IsTensorBase<T>::value;
}  // namespace uvl::concepts
#endif
