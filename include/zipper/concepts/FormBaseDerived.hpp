#if !defined(ZIPPER_CONCEPTS_FORMBASE_DERIVED_HPP)
#define ZIPPER_CONCEPTS_FORMBASE_DERIVED_HPP
#include "ViewDerived.hpp"
#include "ZipperBaseDerived.hpp"

namespace zipper {} // namespace zipper
namespace zipper::concepts {
namespace detail {
template <typename> struct FormBaseDerived : std::false_type {};

template <typename View>
  requires(FormBaseDerived<View>::value)
struct ZipperBaseDerived<View> : std::true_type {};
} // namespace detail

template <typename T>
concept FormBaseDerived =
    (concepts::QualifiedViewDerived<T> &&
     detail::FormBaseDerived<std::remove_cvref_t<T>>::value);

template <typename T>
concept FormLike = (concepts::QualifiedViewDerived<T> &&
                    detail::FormBaseDerived<std::remove_cvref_t<T>>::value);
} // namespace zipper::concepts
#endif
