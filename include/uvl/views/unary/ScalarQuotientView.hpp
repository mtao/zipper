#if !defined(UVL_VIEWS_UNARY_SCALARQUOTIENTVIEW_HPP)
#define UVL_VIEWS_UNARY_SCALARQUOTIENTVIEW_HPP

#include "ScalarOperationView.hpp"

namespace uvl::views {
namespace unary {
template <typename A, concepts::ViewDerived B, bool ScalarOnRight>
using ScalarQuotientView =
    ScalarOperationView<B, std::divides<A>, A, ScalarOnRight>;

}  // namespace unary
}  // namespace uvl::views

#endif
