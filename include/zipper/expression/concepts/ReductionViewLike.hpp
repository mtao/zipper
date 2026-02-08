
#if !defined(ZIPPER_EXPRESSION_CONCEPTS_REDUCTION_VIEW_LIKE_HPP)
#define ZIPPER_EXPRESSION_CONCEPTS_REDUCTION_VIEW_LIKE_HPP
#include <concepts>

#include "zipper/concepts/Expression.hpp"

namespace zipper::views {
template <typename T>
class ViewBase;
}
namespace zipper::views::concepts {
namespace detail {
template <typename T>
struct ReductionViewLike : public std::false_type {};

template <zipper::concepts::Expression T>
    requires(T::extents_type::rank() == 0)
struct ReductionViewLike<T> : public std::true_type {};
}  // namespace detail

template <typename T>
concept ReductionViewLike = detail::ReductionViewLike<std::decay_t<T>>::value;
}  // namespace zipper::views::concepts
#endif
