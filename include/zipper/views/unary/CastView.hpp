#if !defined(ZIPPER_VIEWS_UNARY_CASTVIEW_HPP)
#define ZIPPER_VIEWS_UNARY_CASTVIEW_HPP

#include "OperationView.hpp"

namespace zipper::views::unary {
namespace detail {
template <typename A, typename B>
struct cast {
    constexpr auto operator()(const A& a) const { return static_cast<B>(a); };
};
}  // namespace detail

// converts an underlying view to a new type
template <typename A, concepts::ViewDerived Child>
using CastView = OperationView<
    Child, detail::cast<
               typename zipper::views::detail::ViewTraits<Child>::value_type, A>, true>;

// converts b into a view of type A
template <typename A, concepts::ViewDerived B>
auto cast(const B& b) {
    return CastView<A, B>(b);
}
}  // namespace zipper::views::unary
#endif

