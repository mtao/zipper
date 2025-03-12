#if !defined(UVL_CONCEPTS_MAPPABLEVIEW_DERIVED_HPP)
#define UVL_CONCEPTS_MAPPABLEVIEW_DERIVED_HPP
#include <concepts>
#include <type_traits>

namespace uvl::views {
template <typename T>
class StaticMappedViewBase;
template <typename T>
class DynamicMappedViewBase;
}
namespace uvl::concepts {
template <typename T>
concept StaticMappedViewDerived = std::derived_from<T, uvl::views::StaticMappedViewBase<T>>;
template <typename T>
concept DynamicMappedViewDerived = std::derived_from<T, uvl::views::DynamicMappedViewBase<T>>;

template <typename T>
concept MappedViewDerived = StaticMappedViewDerived<T> || DynamicMappedViewDerived<T>;
}  // namespace uvl::concepts
#endif
