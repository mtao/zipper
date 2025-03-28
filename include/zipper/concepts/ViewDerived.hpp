#if !defined(ZIPPER_CONCEPTS_VIEW_DERIVED_HPP)
#define ZIPPER_CONCEPTS_VIEW_DERIVED_HPP
#include <concepts>

namespace zipper::views {
template <typename T>
class ViewBase;
}
namespace zipper::concepts {

template <typename T>
concept ViewDerived = std::derived_from<T, zipper::views::ViewBase<T>>;
}  // namespace zipper::concepts
#endif
