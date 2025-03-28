#if !defined(ZIPPER_CONCEPTS_SLICEPACKLIKE_HPP)
#define ZIPPER_CONCEPTS_SLICEPACKLIKE_HPP
#include <type_traits>

#include "SliceLike.hpp"

namespace zipper::concepts {
namespace detail {
template <typename... T>
struct slice_pack_like
    : public std::bool_constant<(slice_like<std::decay_t<T>>::value && ...)> {};

}  // namespace detail

template <typename... T>
concept SlicePackLike = detail::slice_pack_like<T...>::value;
}  // namespace zipper::concepts
#endif
