#if !defined(ZIPPER_CONCEPTS_MATRIXBASEDERIVED_HPP)
#define ZIPPER_CONCEPTS_MATRIXBASEDERIVED_HPP
#include "ViewDerived.hpp"
#include "ZipperBaseDerived.hpp"

// namespace zipper
namespace zipper::concepts {
namespace detail {
template <typename> struct MatrixBaseDerived : std::false_type {};

template <typename View>
  requires(MatrixBaseDerived<View>::value)
struct ZipperBaseDerived<View> : std::true_type {};

} // namespace detail

template <typename T>
concept MatrixBaseDerived = detail::MatrixBaseDerived<std::decay_t<T>>;

template <typename T>
concept MatrixLike = (concepts::QualifiedViewDerived<T> &&
                      detail::MatrixBaseDerived<std::remove_cvref_t<T>>::value);
} // namespace zipper::concepts
#endif
