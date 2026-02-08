
#if !defined(ZIPPER_VIEWS_DETAIL_STATIC_EXTENTS_TO_INTEGRAL_SEQUENCE_HPP)
#define ZIPPER_VIEWS_DETAIL_STATIC_EXTENTS_TO_INTEGRAL_SEQUENCE_HPP
#include "zipper/concepts/Extents.hpp"
#include "zipper/types.hpp"
namespace zipper::detail::extents {

// assigns the first parameter into the second parameter
template <typename>
struct static_extents_to_integral_sequence;
template <index_type... Indices>
struct static_extents_to_integral_sequence<zipper::extents<Indices...>> {
    using type = std::integer_sequence<index_type, Indices...>;
};

template <concepts::Extents From>
using static_extents_to_integral_sequence_t =
    static_extents_to_integral_sequence<From>::type;

}  // namespace zipper::detail::extents
#endif
