#if !defined(UVL_VIEWS_DETAIL_VIEW_TYPE_HPP)
#define UVL_VIEWS_DETAIL_VIEW_TYPE_HPP
#include "uvl/concepts/ViewDerived.hpp"
#include "uvl/types.hpp"
namespace uvl::views::detail {
template <typename T>
struct ViewType {
    using type = T::view_type;
};

template <concepts::ViewDerived T>
struct ViewType<T> {
    using type = T;
};

}  // namespace uvl::views::detail
#endif
