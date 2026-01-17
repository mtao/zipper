#if !defined(ZIPPER_CONCEPTS_VIEW_DERIVED_HPP)
#define ZIPPER_CONCEPTS_VIEW_DERIVED_HPP
#include <concepts>

namespace zipper::views {
template <typename T> class ViewBase;
}
namespace zipper::concepts {

/// Every view must be derived from either ViewBase via CRTP, with constness
/// qualification to specify if it is a const type view
template <typename T>
concept UnqualifiedView =
    std::derived_from<T, zipper::views::ViewBase<T>> ||
    std::derived_from<T, zipper::views::ViewBase<const T>>;

template <typename T>
concept View = UnqualifiedView<std::remove_cvref_t<T>>;

} // namespace zipper::concepts
#endif
