#if !defined(UVL_VIEWS_UNARY_SCALARPRODUCTVIEW_HPP)
#define UVL_VIEWS_UNARY_SCALARPRODUCTVIEW_HPP

#include "ScalarOperationView.hpp"

namespace uvl::views {
namespace unary {
template <typename A, concepts::ViewDerived B, bool ScalarOnRight>
using ScalarProductView =
    ScalarOperationView<B, std::multiplies<A>, A, ScalarOnRight>;

template <typename A, concepts::ViewDerived B>
auto scalar_product_view(const A& a, const B& b)
    -> ScalarProductView<A, B, false> {
    return {b, a};
}
template <typename A, concepts::ViewDerived B>
auto scalar_product_view(const B& b, const A& a)
    -> ScalarProductView<A, B, true> {
    return {b, a};
}
}  // namespace unary
}  // namespace uvl::views
#endif
