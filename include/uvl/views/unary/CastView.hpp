#if !defined(UVL_VIEWS_UNARY_CASTVIEW_HPP)
#define UVL_VIEWS_UNARY_CASTVIEW_HPP

#include "OperationView.hpp"

namespace uvl::views::unary {
namespace detail {
template <typename A, typename B>
struct cast {
    constexpr auto operator()(const A& a) const { return static_cast<B>(a); };
};
}  // namespace detail

template <typename A, concepts::ViewDerived Child>
using CastView = OperationView<
    Child, detail::cast<
               typename uvl::views::detail::ViewTraits<Child>::value_type, A>>;

template <typename A, concepts::ViewDerived B>
auto cast(const B& b) {
    return CastView<A, B>(b);
}
}  // namespace uvl::views::unary
#endif

