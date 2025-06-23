#if !defined(ZIPPER_CONCEPTS_VECTORBASE_DERIVED_HPP)
#define ZIPPER_CONCEPTS_VECTORBASE_DERIVED_HPP
#include <concepts>
#include <type_traits>

#include "VectorViewDerived.hpp"

namespace zipper {
template <concepts::ViewDerived T>
class VectorBase;
template <typename T, index_type R>
class Vector;
}  // namespace zipper
namespace zipper::concepts {
namespace detail {
template <typename>
struct IsVector : std::false_type {};
template <typename T, index_type R>
struct IsVector<Vector<T, R>> : std::true_type {};

template <typename>
struct IsVectorBase : std::false_type {};
template <typename T>
struct IsVectorBase<VectorBase<T>> : std::true_type {};

template <typename T>
concept VectorBaseDerived =
    (concepts::VectorViewDerived<T> &&
     std::derived_from<T, zipper::VectorBase<T>>) ||
    detail::IsVector<T>::value || detail::IsVectorBase<T>::value;
}  // namespace detail

template <typename T>
concept VectorBaseDerived = detail::VectorBaseDerived<std::decay_t<T>>;
}  // namespace zipper::concepts
#endif
