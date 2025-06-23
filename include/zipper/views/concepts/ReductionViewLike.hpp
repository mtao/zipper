
#if !defined(ZIPPER_VIEWS_CONCEPTS_VIEW_DERIVED_HPP)
#define ZIPPER_VIEWS_CONCEPTS_VIEW_DERIVED_HPP
#include <concepts>

#include "zipper/concepts/ViewDerived.hpp"

namespace zipper::views {
template <typename T>
class ViewBase;
}
namespace zipper::views::concepts {
namespace detail {
template <typename T>
struct ReductionViewLike : public std::false_type {};

template <zipper::concepts::ViewDerived T>
    requires(T::extents_type::rank() == 0)
struct ReductionViewLike<T> : public std::true_type {};
}  // namespace detail

template <typename T>
concept ReductionViewLike = detail::ReductionViewLike<std::decay_t<T>>::value;
}  // namespace zipper::views::concepts
#endif
