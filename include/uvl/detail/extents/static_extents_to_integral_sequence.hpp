
#if !defined(UVL_VIEWS_DETAIL_STATIC_EXTENTS_TO_INTEGRAL_SEQUENCE_HPP)
#define UVL_VIEWS_DETAIL_STATIC_EXTENTS_TO_INTEGRAL_SEQUENCE_HPP
#include "uvl/concepts/ExtentsType.hpp"
#include "uvl/types.hpp"
namespace uvl::detail::extents {

// assigns the first parameter into the second parameter
template <typename>
struct static_extents_to_integral_sequence;
template <index_type... Indices>
struct static_extents_to_integral_sequence<uvl::extents<Indices...>> {
    using type = std::integer_sequence<index_type, Indices...>;
};

template <concepts::ExtentsType From>
using static_extents_to_integral_sequence_t =
    static_extents_to_integral_sequence<From>::type;

}  // namespace uvl::detail::extents
#endif
