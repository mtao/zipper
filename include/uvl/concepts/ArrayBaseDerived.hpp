

#if !defined(UVL_CONCEPTS_ARRAYBASE_DERIVED_HPP)
#define UVL_CONCEPTS_ARRAYBASE_DERIVED_HPP
#include <concepts>
#include <type_traits>

#include "ViewDerived.hpp"
#include "uvl/types.hpp"

namespace uvl {
template <concepts::ViewDerived T>
class ArrayBase;
template <typename T, index_type... Indices>
    requires(sizeof...(Indices) > 0)
class Array;
}  // namespace uvl
namespace uvl::concepts {
namespace detail {
template <typename>
struct IsArray : std::false_type {};
template <typename T, index_type... Indices>
struct IsArray<Array<T, Indices...>> : std::true_type {};

template <typename>
struct IsArrayBase : std::false_type {};
template <typename T>
struct IsArrayBase<ArrayBase<T>> : std::true_type {};
}  // namespace detail

template <typename T>
concept ArrayBaseDerived =
    (concepts::ViewDerived<T> && std::derived_from<T, uvl::ArrayBase<T>>) ||
    detail::IsArray<T>::value || detail::IsArrayBase<T>::value;
}  // namespace uvl::concepts
#endif
