#if !defined(UVL_CONCEPTS_INDEXPACKLIKE_HPP)
#define UVL_CONCEPTS_INDEXPACKLIKE_HPP
#include <type_traits>

#include "IndexLike.hpp"

namespace uvl::concepts {
namespace detail {
template <typename... T>
struct index_pack_like
    : public std::bool_constant<(index_like<std::decay_t<T>>::value && ...)> {};

}  // namespace detail

template <typename... T>
concept IndexPackLike = detail::index_pack_like<T...>::value;
}  // namespace uvl::concepts
#endif
