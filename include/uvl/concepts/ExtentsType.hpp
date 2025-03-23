#if !defined(UVL_CONCEPTS_EXTENTSTYPE_HPP)
#define UVL_CONCEPTS_EXTENTSTYPE_HPP
#include <type_traits>

#include "uvl/types.hpp"

namespace uvl::concepts {
namespace detail {
template <typename... T>
struct extents_type : public std::false_type {};

template <index_type... Indices>
struct extents_type<extents<Indices...>> : public std::true_type {};

}  // namespace detail

template <typename... T>
concept ExtentsType = detail::extents_type<T...>::value;
}  // namespace uvl::concepts
#endif
