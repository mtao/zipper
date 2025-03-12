#if !defined(UVL_CONCEPTS_VIEW_DERIVED_HPP)
#define UVL_CONCEPTS_VIEW_DERIVED_HPP
#include <concepts>
#include <type_traits>

namespace uvl::views {
template <typename T>
class ViewBase;
}
namespace uvl::concepts {

template <typename T>
concept ViewDerived = std::derived_from<T, uvl::views::ViewBase<T>>;
}  // namespace uvl::concepts
#endif
