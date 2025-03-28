#if !defined(ZIPPER_CONCEPTS_MAPPABLEVIEW_DERIVED_HPP)
#define ZIPPER_CONCEPTS_MAPPABLEVIEW_DERIVED_HPP
#include <concepts>
#include <type_traits>

namespace zipper::views {
template <typename T>
class StaticMappedViewBase;
template <typename T>
class DynamicMappedViewBase;
}
namespace zipper::concepts {
template <typename T>
concept StaticMappedViewDerived = std::derived_from<T, zipper::views::StaticMappedViewBase<T>>;
template <typename T>
concept DynamicMappedViewDerived = std::derived_from<T, zipper::views::DynamicMappedViewBase<T>>;

template <typename T>
concept MappedViewDerived = StaticMappedViewDerived<T> || DynamicMappedViewDerived<T>;
}  // namespace zipper::concepts
#endif
