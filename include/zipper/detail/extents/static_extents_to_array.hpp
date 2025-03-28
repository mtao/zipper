#if !defined(ZIPPER_VIEWS_DETAIL_STATIC_EXTENTS_TO_ARRAY_HPP)
#define ZIPPER_VIEWS_DETAIL_STATIC_EXTENTS_TO_ARRAY_HPP
#include "../sequence_to_array.hpp"
#include "zipper/concepts/ExtentsType.hpp"
#include "zipper/types.hpp"
namespace zipper::detail::extents {

// assigns the first parameter into the second parameter
template <typename>
struct static_extents_to_array;
template <index_type... Indices>
struct static_extents_to_array<zipper::extents<Indices...>>
    : public sequence_to_array<zipper::extents<Indices...>> {};

template <concepts::ExtentsType From>
constexpr static std::array<index_type, From::rank()>
    static_extents_to_array_v = static_extents_to_array<From>::value;

}  // namespace zipper::detail::extents
#endif
