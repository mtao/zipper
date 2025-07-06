#if !defined(ZIPPER_CONCEPTS_SHAPES_HPP)
#define ZIPPER_CONCEPTS_SHAPES_HPP
#include <type_traits>

#include "ExtentsType.hpp"
#include "ZipperBaseDerived.hpp"
#include "zipper/detail/extents/is_compatible.hpp"

namespace zipper {
template <template <concepts::ViewDerived> typename DerivedT,
          concepts::ViewDerived View>
class ZipperBase;
}  // namespace zipper
namespace zipper::concepts {
namespace detail {

template <typename, index_type...>
struct ValidExtents : public std::false_type {};

template <ExtentsType Ext, index_type... Shapes>
struct ValidExtents<Ext, Shapes...> {
    constexpr static bool value =
        zipper::detail::extents::detail::is_compatible<true, Ext>(
            std::integer_sequence<index_type, Shapes...>{},
            std::make_integer_sequence<rank_type, Ext::rank()>{});
};

template <concepts::ViewDerived View, index_type... Shapes>
struct ValidExtents<View, Shapes...>
    : public ValidExtents<typename View::extents_type, Shapes...> {};

template <template <concepts::ViewDerived> typename DerivedT,
          concepts::ViewDerived View, index_type... Shapes>
struct ValidExtents<ZipperBase<DerivedT, View>, Shapes...>
    : public ValidExtents<typename View::extents_type, Shapes...> {};

}  // namespace detail

template <typename T, index_type... Shapes>
concept ValidExtents = detail::ValidExtents<T, Shapes...>::value;
}  // namespace zipper::concepts
#endif
