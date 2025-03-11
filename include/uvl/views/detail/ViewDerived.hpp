#if !defined(UVL_VIEWS_DETAIL_VIEW_DERIVED_HPP)
#define UVL_VIEWS_DETAIL_VIEW_DERIVED_HPP
#include <type_traits>
namespace uvl::views {

template <typename T>
class ViewBase;
template <typename T>
concept ViewDerived = std::derived_from<T, ViewBase<T>>;
}  // namespace uvl::views
#endif
