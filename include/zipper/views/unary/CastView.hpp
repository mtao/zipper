#if !defined(ZIPPER_VIEWS_UNARY_CASTVIEW_HPP)
#define ZIPPER_VIEWS_UNARY_CASTVIEW_HPP

#include "OperationView.hpp"

namespace zipper::views::unary {
namespace detail {
template <typename A, typename B>
struct cast {
    constexpr static auto operator()(const A& a) { return static_cast<B>(a); }
};
}  // namespace detail

// converts an underlying view to a new type
template <typename A, zipper::concepts::QualifiedViewDerived Child>
using CastView = OperationView<
    Child,
    detail::cast<typename zipper::views::detail::ViewTraits<std::decay_t<Child>>::value_type,
                 A>,
    true>;

// converts b into a view of type A
template <typename A, zipper::concepts::QualifiedViewDerived B>
auto cast(B& b) {
    return CastView<A, const B>(b);
}

}  // namespace zipper::views
#endif

