
#if !defined(ZIPPER_CONCEPTS_VIEWACCESSPACK_HPP)
#define ZIPPER_CONCEPTS_VIEWACCESSPACK_HPP
#include <type_traits>

#include "SlicePackLike.hpp"
#include "IndexPackLike.hpp"

namespace zipper::concepts {
namespace detail {

}  // namespace detail
template <typename... T>
concept ViewAccessPack = SlicePackLike<T...> || IndexPackLike<T...>;

}  // namespace zipper::concepts
#endif
