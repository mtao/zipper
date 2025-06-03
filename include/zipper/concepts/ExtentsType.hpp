#if !defined(ZIPPER_CONCEPTS_EXTENTSTYPE_HPP)
#define ZIPPER_CONCEPTS_EXTENTSTYPE_HPP
#include <type_traits>

#include "zipper/types.hpp"

namespace zipper::concepts {
namespace detail {
template <typename... T>
struct extents_type : public std::false_type {};

template <index_type... Indices>
struct extents_type<extents<Indices...>> : public std::true_type {};

}  // namespace detail

template <typename T>
concept ExtentsType = detail::extents_type<T>::value;
}  // namespace zipper::concepts
#endif
