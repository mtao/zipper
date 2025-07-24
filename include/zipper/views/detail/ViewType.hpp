#if !defined(ZIPPER_VIEWS_DETAIL_VIEW_TYPE_HPP)
#define ZIPPER_VIEWS_DETAIL_VIEW_TYPE_HPP
#include "zipper/concepts/ViewDerived.hpp"
#include "zipper/types.hpp"
namespace zipper::views::detail {
template <typename T>
struct ViewType {
    using type = T::view_type;
};

template <zipper::concepts::ViewDerived T>
struct ViewType<T> {
    using type = T;
};

}  // namespace zipper::views::detail
#endif
