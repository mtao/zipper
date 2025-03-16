
#if !defined(UVL_CONCEPTS_VIEWACCESSPACK_HPP)
#define UVL_CONCEPTS_VIEWACCESSPACK_HPP
#include <type_traits>

#include "SlicePackLike.hpp"
#include "IndexPackLike.hpp"

namespace uvl::concepts {
namespace detail {

}  // namespace detail
template <typename... T>
concept ViewAccessPack = SlicePackLike<T...> || IndexPackLike<T...>;

}  // namespace uvl::concepts
#endif
