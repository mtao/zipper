#if !defined(UVL_VIEWS_UNARY_NEGATEVIEW_HPP)
#define UVL_VIEWS_UNARY_NEGATEVIEW_HPP

#include "OperationView.hpp"

namespace uvl::views::unary {

template <concepts::ViewDerived B>
using NegateView = OperationView<
    B, std::negate<typename uvl::views::detail::ViewTraits<B>::value_type> >;

}  // namespace uvl::views::unary
#endif
