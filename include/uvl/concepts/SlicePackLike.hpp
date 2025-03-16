#if !defined(UVL_CONCEPTS_SLICEPACKLIKE_HPP)
#define UVL_CONCEPTS_SLICEPACKLIKE_HPP
#include <type_traits>

#include "SliceLike.hpp"

namespace uvl::concepts {
namespace detail {
template <typename... T>
struct slice_pack_like
    : public std::bool_constant<(slice_like<std::decay_t<T>>::value && ...)> {};

}  // namespace detail

template <typename... T>
concept SlicePackLike = detail::slice_pack_like<T...>::value;
}  // namespace uvl::concepts
#endif
