#if !defined(UVL_VIEWS_DETAIL_STATIC_EXTENTS_TO_ARRAY_HPP)
#define UVL_VIEWS_DETAIL_STATIC_EXTENTS_TO_ARRAY_HPP
#include "../sequence_to_array.hpp"
#include "uvl/concepts/ExtentsType.hpp"
#include "uvl/types.hpp"
namespace uvl::detail::extents {

// assigns the first parameter into the second parameter
template <typename>
struct static_extents_to_array;
template <index_type... Indices>
struct static_extents_to_array<uvl::extents<Indices...>>
    : public sequence_to_array<uvl::extents<Indices...>> {};

template <concepts::ExtentsType From>
constexpr static std::array<index_type, From::rank()>
    static_extents_to_array_v = static_extents_to_array<From>::value;

}  // namespace uvl::detail::extents
#endif
