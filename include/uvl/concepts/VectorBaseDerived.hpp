#if !defined(UVL_CONCEPTS_VECTORBASE_DERIVED_HPP)
#define UVL_CONCEPTS_VECTORBASE_DERIVED_HPP
#include <concepts>
#include <type_traits>

#include "VectorViewDerived.hpp"

namespace uvl {
template <concepts::ViewDerived T>
class VectorBase;
template <typename T, index_type R>
class Vector;
}  // namespace uvl
namespace uvl::concepts {
namespace detail {
template <typename>
struct IsVector : std::false_type {};
template <typename T, index_type R>
struct IsVector<Vector<T, R>> : std::true_type {};

template <typename>
struct IsVectorBase : std::false_type {};
template <typename T>
struct IsVectorBase<VectorBase<T>> : std::true_type {};
}  // namespace detail

template <typename T>
concept VectorBaseDerived =
    (concepts::VectorViewDerived<T> &&
     std::derived_from<T, uvl::VectorBase<T>>) ||
    detail::IsVector<T>::value || detail::IsVectorBase<T>::value;
}  // namespace uvl::concepts
#endif
