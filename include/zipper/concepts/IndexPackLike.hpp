#if !defined(ZIPPER_CONCEPTS_INDEXPACKLIKE_HPP)
#define ZIPPER_CONCEPTS_INDEXPACKLIKE_HPP
#include <type_traits>

#include "IndexLike.hpp"

namespace zipper::concepts {
namespace detail {
template <typename... T>
struct index_pack_like
    : public std::bool_constant<(index_like<std::decay_t<T>>::value && ...)> {};

}  // namespace detail

template <typename... T>
concept IndexPackLike = detail::index_pack_like<T...>::value;
}  // namespace zipper::concepts
#endif
