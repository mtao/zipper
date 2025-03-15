#if !defined(UVL_VIEWS_UNARY_ABSOLUTEVIEW_HPP)
#define UVL_VIEWS_UNARY_ABSOLUTEVIEW_HPP

#include "OperationView.hpp"
namespace uvl::views::unary {
namespace detail {
template <typename A>
struct abs {
    constexpr A operator()(const A& a) const { return std::abs(a); };
};
}  // namespace detail

template <concepts::ViewDerived Child>
using AbsView = OperationView<
    Child,
    detail::abs<typename uvl::views::detail::ViewTraits<Child>::value_type>>;

}  // namespace uvl::views::unary
#endif
