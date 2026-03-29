#if !defined(ZIPPER_DETAIL_VIEWPROPAGATING_HPP)
#define ZIPPER_DETAIL_VIEWPROPAGATING_HPP

#include <concepts>
#include <type_traits>

namespace zipper::detail {

/// Detects whether a type has `is_view_propagating == true` in its traits.
///
/// View-propagating expressions (created via ref()) are stored by value
/// even from lvalue context in `member_child_storage_t` and
/// `expression_storage_t`, so that derived views are also Returnable.
template <typename T>
concept ViewPropagating = requires {
  { std::decay_t<T>::is_view_propagating } -> std::convertible_to<bool>;
} && std::decay_t<T>::is_view_propagating;

} // namespace zipper::detail

#endif
