#if !defined(ZIPPER_CONCEPTS_SHAPES_HPP)
#define ZIPPER_CONCEPTS_SHAPES_HPP
#include <type_traits>

#include "Extents.hpp"
#include "Zipper.hpp"
#include "zipper/utils/extents/is_compatible.hpp"

namespace zipper {
template <template <concepts::QualifiedExpression> typename DerivedT,
          concepts::QualifiedExpression Expr>
class ZipperBase;
}  // namespace zipper
namespace zipper::concepts {
namespace detail {

template <typename T, index_type... Shapes>
struct ValidExtents {
    constexpr static bool value =
        zipper::utils::extents::detail::is_compatible<true, typename T::extents_type>(
            std::integer_sequence<index_type, Shapes...>{},
            std::make_integer_sequence<rank_type, T::extents_type::rank()>{});
};

template <Extents Ext, index_type... Shapes>
struct ValidExtents<Ext, Shapes...> {
    constexpr static bool value =
        zipper::utils::extents::detail::is_compatible<true, Ext>(
            std::integer_sequence<index_type, Shapes...>{},
            std::make_integer_sequence<rank_type, Ext::rank()>{});
};

}  // namespace detail

template <typename T, index_type... Shapes>
concept ValidExtents = detail::ValidExtents<T, Shapes...>::value;
}  // namespace zipper::concepts
#endif
