#if !defined(ZIPPER_CONCEPTS_VECTORBASEDERIVED_HPP)
#define ZIPPER_CONCEPTS_VECTORBASEDERIVED_HPP
#include <concepts>
#include <type_traits>

#include "VectorViewDerived.hpp"

namespace zipper {
template <concepts::QualifiedViewDerived T> class VectorBase;
template <typename T, index_type R> class Vector;
} // namespace zipper
namespace zipper::concepts {
namespace detail {
template <typename> struct IsVector : std::false_type {};
template <typename T, index_type R>
struct IsVector<Vector<T, R>> : std::true_type {};

template <typename> struct VectorBaseDerived : std::false_type {};
template <typename T>
struct VectorBaseDerived<VectorBase<T>> : std::true_type {};

template <typename T>
concept VectorBaseDerived =
    (concepts::VectorViewDerived<T> &&
     std::derived_from<T, zipper::VectorBase<T>>) ||
    detail::IsVector<T>::value || detail::VectorBaseDerived<T>::value;
} // namespace detail

template <typename T>
concept VectorBaseDerived = detail::VectorBaseDerived<std::decay_t<T>>;
template <typename T>
concept VectorLike = (concepts::QualifiedViewDerived<T> &&
                      detail::VectorBaseDerived<std::remove_cvref_t<T>>::value);
} // namespace zipper::concepts
#endif
