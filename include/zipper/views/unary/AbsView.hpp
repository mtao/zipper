#if !defined(ZIPPER_VIEWS_UNARY_ABSOLUTEVIEW_HPP)
#define ZIPPER_VIEWS_UNARY_ABSOLUTEVIEW_HPP

#include "OperationView.hpp"
namespace zipper::views::unary {
namespace detail {
template <typename A>
struct abs {
    constexpr A operator()(const A& a) const { return std::abs(a); };
};
}  // namespace detail

template <concepts::ViewDerived Child>
using AbsView = OperationView<
    Child,
    detail::abs<typename zipper::views::detail::ViewTraits<Child>::value_type>>;

}  // namespace zipper::views::unary
#endif
