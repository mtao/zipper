#if !defined(ZIPPER_CONCEPTS_ARRAYBASEDERIVED_HPP)
#define ZIPPER_CONCEPTS_ARRAYBASEDERIVED_HPP
#include "ViewDerived.hpp"
#include "ZipperBaseDerived.hpp"

namespace zipper::concepts {
namespace detail {

template <typename> struct ArrayBaseDerived : std::false_type {};

template <typename View>
  requires(ArrayBaseDerived<View>::value)
struct ZipperBaseDerived<View> : std::true_type {};
} // namespace detail

template <typename T>
concept ArrayBaseDerived =
    (concepts::QualifiedViewDerived<T> &&
     detail::ArrayBaseDerived<std::remove_cvref_t<T>>::value);
template <typename T>
concept ArrayLike = (concepts::QualifiedViewDerived<T> &&
                     detail::ArrayBaseDerived<std::remove_cvref_t<T>>::value);
} // namespace zipper::concepts
#endif
